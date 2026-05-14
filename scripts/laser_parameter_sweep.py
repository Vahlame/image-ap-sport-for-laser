#!/usr/bin/env python3
"""
Barrido reproducible de parámetros (umbral / contraste / brillo) sobre una misma imagen.

Genera:
  - sweep.sqlite  — tabla `runs` con parámetros + métricas + ruta del PNG
  - sweep_manifest.jsonl — una línea JSON por corrida (fácil de grep)
  - sweep_XXXX.png — salida 1 canal (blanco/negro) misma lógica que el preview web

Uso típico (muchas pruebas, tarda según tamaño y N):

  pip install -e .
  python scripts/laser_parameter_sweep.py --input foto.png --out runs/exp1 --n 400 --seed 42 --max-side 900

El modo por defecto no apuesta a un único umbral: reserva una escalera de thresholds,
incluye umbrales guiados por histograma/Otsu y completa con combinaciones estratificadas.
Luego abrís la carpeta, filtrás por `source`/`white_ratio` en SQLite o mirás visualmente
cuál se acerca al resultado deseado.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Params:
    threshold: int
    contrast: float
    brightness: float
    source: str
    sequence: int


@dataclass
class RunResult:
    id: int
    threshold: int
    contrast: float
    brightness: float
    source: str
    sequence: int
    white_ratio: float
    mean_gray: float
    output_file: str
    seconds: float


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def preview_threshold(gray: np.ndarray, threshold: float, contrast: float, brightness: float) -> np.ndarray:
    """Misma fórmula que `web/src/lib/laserPreview.ts` (preview local)."""
    y = gray.copy()
    y = (y - 128.0) * contrast + 128.0 + brightness
    y = np.clip(y, 0.0, 255.0)
    return np.where(y >= threshold, 255, 0).astype(np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    """Umbral global Otsu sobre gris 0..255, sin depender de OpenCV/skimage."""
    clipped = np.clip(gray, 0, 255).astype(np.uint8)
    hist = np.bincount(clipped.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    bins = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * bins)
    sum_total = sum_bg[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.zeros_like(bins)
    mean_fg = np.zeros_like(bins)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between))


def unique_thresholds(values: list[float] | np.ndarray) -> list[int]:
    rounded = np.clip(np.round(values), 1, 254).astype(int)
    return sorted({int(v) for v in rounded})


def image_threshold_ladder(gray: np.ndarray, steps: int) -> list[int]:
    """
    Umbrales candidatos con dos familias:
    - lineal: no deja huecos grandes aunque la imagen tenga poco rango tonal
    - histograma: concentra pruebas donde realmente hay datos de la imagen
    """
    steps = max(4, int(steps))
    linear = np.linspace(16, 240, steps, dtype=np.float64)
    quantile_points = np.linspace(0.03, 0.97, steps)
    hist = np.quantile(gray, quantile_points)
    anchors = np.array([32, 64, 96, 128, 160, 192, 224, otsu_threshold(gray)], dtype=np.float64)
    return unique_thresholds(np.concatenate([linear, hist, anchors]))


def stratified_params(rng: np.random.Generator, n: int, start_sequence: int = 1) -> list[Params]:
    """
    Muestras ordenadas en el espacio pero con componente aleatorio:
    - threshold: n valores repartidos en [t_min, t_max] + jitter
    - contrast / brightness: permutación independiente para no correlacionar todo
    """
    t_lo, t_hi = 12, 243
    c_lo, c_hi = 0.52, 2.48
    b_lo, b_hi = -58.0, 58.0

    # Posiciones estratificadas en el eje principal (umbral)
    base = np.linspace(t_lo, t_hi, n, dtype=np.float64)
    jitter = rng.uniform(-4.0, 4.0, size=n)
    thresholds = np.clip(np.round(base + jitter), 1, 254).astype(int)

    contrasts = rng.uniform(c_lo, c_hi, size=n)
    brightnesses = rng.uniform(b_lo, b_hi, size=n)
    # Des-correlacionar un poco: permutar contrastes respecto a thresholds
    perm = rng.permutation(n)
    contrasts = contrasts[perm]

    return [
        Params(int(t), float(c), float(br), "stratified_jitter", start_sequence + i)
        for i, (t, c, br) in enumerate(zip(thresholds, contrasts, brightnesses))
    ]


def smart_params(
    gray: np.ndarray,
    rng: np.random.Generator,
    n: int,
    threshold_steps: int,
) -> list[Params]:
    """
    Genera un sweep que prioriza comparar umbrales:
    1. Escalera amplia con contraste/brillo neutro.
    2. Micro-grid alrededor de Otsu, mediana y cuartiles para no perder detalles finos.
    3. Relleno estratificado con contraste/brillo variados.
    """
    if n <= 0:
        return []

    params: list[Params] = []
    seen: set[tuple[int, float, float]] = set()

    def add(threshold: int, contrast: float, brightness: float, source: str) -> None:
        key = (int(threshold), round(float(contrast), 4), round(float(brightness), 4))
        if key in seen or len(params) >= n:
            return
        seen.add(key)
        params.append(Params(key[0], key[1], key[2], source, len(params) + 1))

    for threshold in image_threshold_ladder(gray, threshold_steps):
        add(threshold, 1.0, 0.0, "threshold_ladder")

    anchors = unique_thresholds(
        np.array(
            [
                otsu_threshold(gray),
                np.quantile(gray, 0.25),
                np.quantile(gray, 0.50),
                np.quantile(gray, 0.75),
            ],
            dtype=np.float64,
        )
    )
    for threshold in anchors:
        for contrast in (0.75, 1.0, 1.25, 1.6, 2.0):
            for brightness in (-24.0, -12.0, 0.0, 12.0, 24.0):
                add(threshold, contrast, brightness, "histogram_anchor_grid")

    while len(params) < n:
        remaining = n - len(params)
        before = len(params)
        for par in stratified_params(rng, remaining, start_sequence=len(params) + 1):
            add(par.threshold, par.contrast, par.brightness, par.source)
        if len(params) == before:
            break

    return params


def make_thumbnail(src: Path, dst: Path, thumb_side: int) -> None:
    with Image.open(src) as im:
        thumb = im.convert("RGB")
        thumb.thumbnail((thumb_side, thumb_side), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (thumb_side, thumb_side), (15, 23, 42))
        x = (thumb_side - thumb.width) // 2
        y = (thumb_side - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        canvas.save(dst, optimize=True)


def write_contact_sheet(out_dir: Path, rows: list[RunResult], thumb_side: int) -> Path:
    cols = 5
    label_h = 58
    pad = 14
    cell_w = thumb_side + pad * 2
    cell_h = thumb_side + label_h + pad * 2
    rows_count = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_w, rows_count * cell_h), (8, 13, 23))
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(rows):
        x0 = (idx % cols) * cell_w + pad
        y0 = (idx // cols) * cell_h + pad
        with Image.open(out_dir / row.output_file) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((thumb_side, thumb_side), Image.Resampling.NEAREST)
            sheet.paste(thumb, (x0 + (thumb_side - thumb.width) // 2, y0 + (thumb_side - thumb.height) // 2))
        if draw is not None:
            label = f"#{row.id:03d} thr {row.threshold} w {row.white_ratio:.2f}\n{row.source}"
            draw.text((x0, y0 + thumb_side + 8), label, fill=(226, 232, 240))

    path = out_dir / "contact_sheet.png"
    sheet.save(path, optimize=True)
    return path


def write_html_report(
    out_dir: Path,
    rows: list[RunResult],
    input_name: str,
    image_size: tuple[int, int],
    elapsed: float,
    thumb_side: int,
) -> Path:
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)
    for row in rows:
        make_thumbnail(out_dir / row.output_file, thumbs_dir / row.output_file, thumb_side)

    sources = sorted({row.source for row in rows})
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card" data-source="{escape(row.source)}" data-threshold="{row.threshold}">
              <a href="{escape(row.output_file)}" target="_blank" title="Abrir PNG individual">
                <img src="thumbs/{escape(row.output_file)}" alt="Resultado {row.id}" loading="lazy">
              </a>
              <div class="meta">
                <strong>#{row.id:04d}</strong>
                <span class="pill">{escape(row.source)}</span>
                <span>thr <b>{row.threshold}</b></span>
                <span>c {row.contrast:.2f}</span>
                <span>b {row.brightness:+.1f}</span>
                <span>white <b>{row.white_ratio:.3f}</b></span>
              </div>
            </article>
            """
        )

    buttons = ['<button type="button" class="filter active" data-source="all">Todos</button>']
    buttons.extend(
        f'<button type="button" class="filter" data-source="{escape(source)}">{escape(source)}</button>'
        for source in sources
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sweep láser — {escape(input_name)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #070b12;
      --panel: #0f172a;
      --muted: #8b9cb3;
      --text: #e5edf7;
      --accent: #38bdf8;
      --border: rgba(148, 163, 184, 0.22);
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%), var(--bg);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 18px 22px;
      background: rgba(7, 11, 18, 0.88);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.25rem; }}
    .summary, .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .summary span {{ color: var(--muted); font-size: 0.9rem; }}
    .controls {{ margin-top: 12px; }}
    button, a.sheet {{
      border: 1px solid var(--border);
      background: #111c30;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      text-decoration: none;
      font-weight: 700;
    }}
    button.active, a.sheet {{ border-color: var(--accent); color: #dff7ff; }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 14px;
      padding: 18px;
    }}
    .card {{
      background: rgba(15, 23, 42, 0.86);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}
    .card img {{
      width: 100%;
      display: block;
      image-rendering: pixelated;
      background: #020617;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 5px 8px;
      padding: 10px;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .meta strong, .meta b {{ color: var(--text); }}
    .pill {{
      grid-column: 1 / -1;
      color: var(--accent);
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>Sweep láser — resultados visibles y separables</h1>
    <div class="summary">
      <span>Input: <b>{escape(input_name)}</b></span>
      <span>{image_size[0]}×{image_size[1]} px</span>
      <span>{len(rows)} PNGs</span>
      <span>{elapsed:.2f}s</span>
      <span>DB: sweep.sqlite</span>
    </div>
    <div class="controls">
      {''.join(buttons)}
      <a class="sheet" href="contact_sheet.png" target="_blank">Abrir hoja de contacto</a>
    </div>
  </header>
  <main id="grid">
    {''.join(cards)}
  </main>
  <script>
    const buttons = [...document.querySelectorAll('.filter')];
    const cards = [...document.querySelectorAll('.card')];
    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        for (const b of buttons) b.classList.remove('active');
        button.classList.add('active');
        const source = button.dataset.source;
        for (const card of cards) {{
          card.classList.toggle('hidden', source !== 'all' && card.dataset.source !== source);
        }}
      }});
    }}
  </script>
</body>
</html>
"""
    report_path = out_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def prepare_output_dir(out_dir: Path, resume: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return

    for pattern in ("sweep_*.png", "sweep.sqlite", "sweep_manifest.jsonl", "index.html", "contact_sheet.png"):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()

    thumbs = out_dir / "thumbs"
    if thumbs.is_dir():
        for path in thumbs.glob("*.png"):
            if path.is_file():
                path.unlink()
        try:
            thumbs.rmdir()
        except OSError:
            pass


def ensure_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threshold INTEGER NOT NULL,
            contrast REAL NOT NULL,
            brightness REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'unknown',
            sequence INTEGER NOT NULL DEFAULT 0,
            white_ratio REAL NOT NULL,
            mean_gray REAL NOT NULL,
            output_file TEXT NOT NULL,
            seconds REAL NOT NULL,
            created TEXT NOT NULL
        )
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "source" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
    if "sequence" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN sequence INTEGER NOT NULL DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_source_threshold ON runs(source, threshold)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_white_ratio ON runs(white_ratio)")
    conn.commit()
    return conn


def main() -> int:
    p = argparse.ArgumentParser(description="Barrido de parámetros preview láser (SQLite + PNGs)")
    p.add_argument("--input", required=True, type=Path, help="Imagen RGB de entrada (jpg/png/webp)")
    p.add_argument("--out", required=True, type=Path, help="Carpeta de salida (se crea)")
    p.add_argument("--n", type=int, default=200, help="Cantidad de combinaciones")
    p.add_argument("--seed", type=int, default=42, help="Semilla RNG (reproducible)")
    p.add_argument("--max-side", type=int, default=0, help="Si >0, redimensiona manteniendo aspecto (LANCZOS)")
    p.add_argument(
        "--mode",
        choices=("smart", "stratified"),
        default="smart",
        help="smart = escalera de umbrales + histograma + relleno; stratified = comportamiento anterior",
    )
    p.add_argument(
        "--threshold-steps",
        type=int,
        default=48,
        help="Cantidad base de umbrales para la escalera inteligente",
    )
    p.add_argument("--thumb-side", type=int, default=180, help="Tamaño de miniatura para index.html/contact_sheet.png")
    p.add_argument("--no-report", action="store_true", help="No generar index.html, miniaturas ni contact_sheet.png")
    p.add_argument("--resume", action="store_true", help="Conserva runs previos en la misma carpeta/SQLite")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"No existe: {args.input}", file=sys.stderr)
        return 2

    prepare_output_dir(args.out, args.resume)
    db_path = args.out / "sweep.sqlite"
    manifest_path = args.out / "sweep_manifest.jsonl"

    im = Image.open(args.input).convert("RGB")
    if args.max_side > 0:
        im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)

    rgb = np.array(im)
    gray = rgb_to_gray(rgb)
    rng = np.random.default_rng(args.seed)
    if args.mode == "smart":
        params_list = smart_params(gray, rng, args.n, args.threshold_steps)
    else:
        params_list = stratified_params(rng, args.n)

    conn = ensure_db(db_path)
    t0_all = time.perf_counter()

    with manifest_path.open("w", encoding="utf-8") as mf:
        results: list[RunResult] = []
        for i, par in enumerate(params_list, start=1):
            t0 = time.perf_counter()
            out = preview_threshold(gray, par.threshold, par.contrast, par.brightness)
            white_ratio = float(np.mean(out == 255))
            mean_gray = float(np.mean(gray))

            fname = f"sweep_{i:04d}.png"
            fpath = args.out / fname
            Image.fromarray(out, mode="L").save(fpath, optimize=True)

            elapsed = time.perf_counter() - t0
            row = {
                "id": i,
                **asdict(par),
                "white_ratio": white_ratio,
                "mean_gray": mean_gray,
                "output_file": fname,
                "seconds": elapsed,
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(RunResult(**row))

            conn.execute(
                """
                INSERT INTO runs (
                    threshold, contrast, brightness, source, sequence,
                    white_ratio, mean_gray, output_file, seconds, created
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    par.threshold,
                    par.contrast,
                    par.brightness,
                    par.source,
                    par.sequence,
                    white_ratio,
                    mean_gray,
                    fname,
                    elapsed,
                ),
            )

            if i % 50 == 0 or i == len(params_list):
                conn.commit()
                print(f"  {i}/{len(params_list)} … último white_ratio={white_ratio:.3f}", flush=True)

    conn.commit()
    conn.close()

    total = time.perf_counter() - t0_all
    print(f"Listo: {args.out}")
    print(f"  SQLite: {db_path}")
    print(f"  JSONL:  {manifest_path}")
    print(f"  Tiempo total: {total:.1f}s")
    if not args.no_report:
        contact_sheet = write_contact_sheet(args.out, results, args.thumb_side)
        report = write_html_report(args.out, results, args.input.name, im.size, total, args.thumb_side)
        print(f"  Reporte: {report}")
        print(f"  Hoja:    {contact_sheet}")

    # Sugerencia: top 5 más “balanceados” (cerca de 50% blanco)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT threshold, contrast, brightness, source, white_ratio, output_file
        FROM runs
        ORDER BY ABS(white_ratio - 0.5) ASC
        LIMIT 8
        """
    ).fetchall()
    conn.close()
    print("\nCandidatos cercanos a 50% blanco (solo heurística):")
    for r in rows:
        print(f"  thr={r[0]:3d}  c={r[1]:.2f}  b={r[2]:+.1f}  {r[3]:21s}  white={r[4]:.3f}  -> {r[5]}")

    conn = sqlite3.connect(db_path)
    threshold_rows = conn.execute(
        """
        SELECT threshold, white_ratio, output_file
        FROM runs
        WHERE source = 'threshold_ladder'
        ORDER BY threshold ASC
        LIMIT 12
        """
    ).fetchall()
    conn.close()
    if threshold_rows:
        print("\nMuestra de escalera de umbrales (contraste=1, brillo=0):")
        for threshold, white_ratio, output_file in threshold_rows:
            print(f"  thr={threshold:3d}  white={white_ratio:.3f}  -> {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
