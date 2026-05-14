#!/usr/bin/env python3
"""
Ejecuta laser_target_match.py sobre varias imágenes y crea un índice comparativo.

Uso:
  python scripts/laser_batch_match.py --target ref.png --input img1.png --out runs/batch --n 800
  # workers por imagen: por defecto todos los núcleos en el hijo; varias imágenes a la vez:
  python scripts/laser_batch_match.py ... --jobs 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BatchRow:
    name: str
    input_file: str
    run_dir: str
    best_file: str
    algorithm: str
    score: float
    threshold: int
    contrast: float
    brightness: float
    gamma: float
    white_ratio: float


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "image"


def collect_inputs(inputs: list[Path], input_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    if input_dir is not None:
        if not input_dir.is_dir():
            raise FileNotFoundError(input_dir)
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            paths.extend(sorted(input_dir.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def child_match_workers(batch_jobs: int, cli_workers: int) -> int:
    """Workers por proceso hijo. Evita sobresuscribir cuando --jobs > 1."""
    cpus = os.cpu_count() or 2
    if batch_jobs <= 1:
        return cli_workers if cli_workers > 0 else 0
    if cli_workers > 0:
        return max(1, cli_workers // batch_jobs)
    return max(1, cpus // batch_jobs)


def run_single_batch_item(
    index: int,
    total: int,
    image_path: Path,
    args: argparse.Namespace,
    script: Path,
    batch_jobs: int,
) -> tuple[int, BatchRow]:
    run_name = f"{index:02d}-{slugify(image_path.stem)}"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("batch_input.json").write_text(
        json.dumps({"name": image_path.stem, "input_file": str(image_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    worker_flag = child_match_workers(batch_jobs, args.workers)
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(image_path),
        "--target",
        str(args.target),
        "--out",
        str(run_dir),
        "--n",
        str(args.n),
        "--max-side",
        str(args.max_side),
        "--top-report",
        str(args.top_report),
        "--workers",
        str(worker_flag),
    ]
    print(f"[{index}/{total}] {image_path} -> {run_dir} (workers/hijo={worker_flag})", flush=True)
    subprocess.run(cmd, check=True)
    return index, best_row(run_dir)


def best_row(run_dir: Path) -> BatchRow:
    db_path = run_dir / "match.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT algorithm, threshold, contrast, brightness, gamma, score, white_ratio, output_file
            FROM matches
            ORDER BY score ASC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Sin resultados en {db_path}")
    manifest = run_dir / "batch_input.json"
    info = json.loads(manifest.read_text(encoding="utf-8"))
    return BatchRow(
        name=info["name"],
        input_file=info["input_file"],
        run_dir=run_dir.name,
        best_file=str(Path(run_dir.name) / row["output_file"]).replace("\\", "/"),
        algorithm=str(row["algorithm"]),
        score=float(row["score"]),
        threshold=int(row["threshold"]),
        contrast=float(row["contrast"]),
        brightness=float(row["brightness"]),
        gamma=float(row["gamma"]),
        white_ratio=float(row["white_ratio"]),
    )


def write_index(out_dir: Path, rows: list[BatchRow]) -> None:
    cards = []
    for row in sorted(rows, key=lambda item: item.score):
        cards.append(
            f"""
            <article class="card">
              <a href="{escape(row.run_dir)}/index.html"><img src="{escape(row.best_file)}" alt="{escape(row.name)}"></a>
              <div class="meta">
                <strong>{escape(row.name)}</strong>
                <span>score <b>{row.score:.4f}</b></span>
                <span>{escape(row.algorithm)}</span>
                <span>thr {row.threshold}</span>
                <span>c {row.contrast:.2f}</span>
                <span>b {row.brightness:+.0f}</span>
                <span>g {row.gamma:.2f}</span>
                <span>white {row.white_ratio:.3f}</span>
              </div>
            </article>
            """
        )
    table_rows = "\n".join(
        f"<tr><td><a href='{escape(row.run_dir)}/index.html'>{escape(row.name)}</a></td><td>{row.score:.4f}</td><td>{escape(row.algorithm)}</td><td>{row.threshold}</td><td>{row.contrast:.2f}</td><td>{row.brightness:+.0f}</td><td>{row.gamma:.2f}</td><td>{row.white_ratio:.3f}</td></tr>"
        for row in sorted(rows, key=lambda item: item.score)
    )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Batch match láser</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070b12; --panel:#0f172a; --muted:#8b9cb3; --text:#e5edf7; --accent:#38bdf8; --border:rgba(148,163,184,.22); }}
    body {{ margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:20px 24px; border-bottom:1px solid var(--border); background:#08111f; }}
    h1 {{ margin:0 0 8px; }}
    p {{ margin:0; color:var(--muted); }}
    main {{ padding:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px; }}
    .card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
    .card img {{ width:100%; display:block; image-rendering:pixelated; background:#020617; }}
    .meta {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 8px; padding:10px; font-size:.84rem; color:var(--muted); }}
    .meta strong {{ grid-column:1/-1; color:var(--text); }}
    b {{ color:var(--text); }}
    table {{ width:100%; border-collapse:collapse; margin-top:22px; background:var(--panel); border:1px solid var(--border); }}
    th,td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; font-size:.9rem; }}
    th {{ color:var(--accent); }}
    a {{ color:#dff7ff; }}
  </style>
</head>
<body>
  <header>
    <h1>Batch match láser</h1>
    <p>{len(rows)} imágenes procesadas. Cada tarjeta abre la galería completa de esa imagen.</p>
  </header>
  <main>
    <section class="grid">{''.join(cards)}</section>
    <table>
      <thead><tr><th>Imagen</th><th>Score</th><th>Algoritmo</th><th>Thr</th><th>Contraste</th><th>Brillo</th><th>Gamma</th><th>White</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""
    out_dir.joinpath("index.html").write_text(html, encoding="utf-8")
    out_dir.joinpath("batch_summary.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best = min(rows, key=lambda item: item.score)
    report_lines = [
        "# Batch Best Match Report",
        "",
        f"- Best input: `{best.name}`",
        f"- Best image: `{best.best_file}`",
        f"- Score: `{best.score:.6f}`",
        f"- Algorithm: `{best.algorithm}`",
        f"- Threshold: `{best.threshold}`",
        f"- Contrast: `{best.contrast:.4f}`",
        f"- Brightness: `{best.brightness:+.4f}`",
        f"- Gamma: `{best.gamma:.4f}`",
        f"- White ratio: `{best.white_ratio:.6f}`",
        "",
        "## Ranking",
        "",
    ]
    for index, row in enumerate(sorted(rows, key=lambda item: item.score), start=1):
        report_lines.append(
            f"{index}. `{row.name}` score `{row.score:.6f}` | `{row.algorithm}` "
            f"thr `{row.threshold}` c `{row.contrast:.3f}` b `{row.brightness:+.1f}` g `{row.gamma:.3f}`"
        )
    out_dir.joinpath("best_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch target-match para varias imágenes")
    parser.add_argument("--target", required=True, type=Path, help="Imagen objetivo/reference")
    parser.add_argument("--input", action="append", type=Path, default=[], help="Imagen de entrada; puede repetirse")
    parser.add_argument("--input-dir", type=Path, default=None, help="Carpeta con imágenes")
    parser.add_argument("--out", required=True, type=Path, help="Carpeta raíz de salida batch")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--max-side", type=int, default=240)
    parser.add_argument("--top-report", type=int, default=120)
    parser.add_argument("--workers", type=int, default=0, help="Procesos en cada laser_target_match; 0 = todos los núcleos en el hijo")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Cuántas imágenes procesar en paralelo (cada una lanza su propio Python). "
        "Si >1 y workers=0, reparte núcleos entre hijos (cpus/jobs por imagen).",
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"No existe target: {args.target}", file=sys.stderr)
        return 2
    try:
        inputs = collect_inputs(args.input, args.input_dir)
    except FileNotFoundError as exc:
        print(f"No existe input: {exc}", file=sys.stderr)
        return 2
    if not inputs:
        print("Agrega al menos un --input o --input-dir", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("laser_target_match.py")
    rows: list[BatchRow] = []
    total = len(inputs)
    batch_jobs = max(1, args.jobs)

    if batch_jobs == 1:
        for index, image_path in enumerate(inputs, start=1):
            _, row = run_single_batch_item(index, total, image_path, args, script, batch_jobs=1)
            rows.append(row)
            write_index(args.out, rows)
            current_best = min(rows, key=lambda item: item.score)
            print(
                f"Mejor batch actual: {current_best.name} -> {current_best.best_file} "
                f"score={current_best.score:.6f}",
                flush=True,
            )
    else:
        print(
            f"Batch paralelo: jobs={batch_jobs}, workers/hijo segun CPUs "
            f"(total logico ~ {os.cpu_count() or '?'} nucleos)",
            flush=True,
        )
        indexed_rows: list[tuple[int, BatchRow]] = []
        with ThreadPoolExecutor(max_workers=batch_jobs) as pool:
            futures = [
                pool.submit(run_single_batch_item, index, total, image_path, args, script, batch_jobs)
                for index, image_path in enumerate(inputs, start=1)
            ]
            for fut in as_completed(futures):
                indexed_rows.append(fut.result())
        rows = [row for _, row in sorted(indexed_rows, key=lambda item: item[0])]
        write_index(args.out, rows)
        current_best = min(rows, key=lambda item: item.score)
        print(
            f"Mejor batch: {current_best.name} -> {current_best.best_file} "
            f"score={current_best.score:.6f}",
            flush=True,
        )

    print(f"Listo batch: {args.out / 'index.html'}")
    print(f"Best report: {args.out / 'best_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
