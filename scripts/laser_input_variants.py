#!/usr/bin/env python3
"""
Genera variantes sistematicas de una imagen antes del dither final.

La idea es barata y agresiva: explorar crop/rotacion/tono primero, pre-rankear
contra el target con densidad y bordes, y pasar solo las mejores al batch match.

Modo `explore`: ~10k+ specs (crop/offset/rotación/brillo/contraste/local/sharp); usar `--limit` 100–200+.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class VariantSpec:
    scale: float
    offset_x: float
    offset_y: float
    rotation: float
    brightness: float
    contrast: float
    local_contrast: float
    sharpness: float
    source: str


@dataclass
class VariantResult:
    id: int
    output_file: str
    pre_score: float
    scale: float
    offset_x: float
    offset_y: float
    rotation: float
    brightness: float
    contrast: float
    local_contrast: float
    sharpness: float
    source: str


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def rgb_to_gray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"), dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def otsu_threshold(gray: np.ndarray) -> int:
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


def otsu_threshold_clipped(gray: np.ndarray, low_pct: float = 1.5, high_pct: float = 98.5) -> int:
    """Otsu sobre histograma recortado en percentiles (menos sensible a colas / ruido)."""
    lo = float(np.percentile(gray, low_pct))
    hi = float(np.percentile(gray, high_pct))
    if hi <= lo + 1e-6:
        return otsu_threshold(gray)
    clipped = np.clip((gray - lo) / (hi - lo) * 255.0, 0.0, 255.0)
    return otsu_threshold(clipped)


def li_threshold(gray: np.ndarray, max_iter: int = 64) -> int:
    """Umbral de Li (iterativo entre medias bajo/alto); distinto de Otsu."""
    t = float(np.mean(gray))
    for _ in range(max_iter):
        low = gray[gray <= t]
        high = gray[gray > t]
        if low.size == 0 or high.size == 0:
            break
        t_new = 0.5 * (float(np.mean(low)) + float(np.mean(high)))
        if abs(t_new - t) < 0.25:
            break
        t = t_new
    return int(np.clip(round(t), 1, 254))


def triangle_threshold(gray: np.ndarray) -> int:
    """Método triángulo (Zack) en histograma 256 bins; útil cuando un pico domina."""
    h = np.bincount(np.clip(gray.astype(np.int64), 0, 255).ravel(), minlength=256).astype(np.float64)
    total = h.sum()
    if total <= 0:
        return 128
    peak = int(np.argmax(h))
    if h[peak] <= 0:
        return 128
    flip = peak < 128
    if flip:
        h = h[::-1].copy()
        peak = int(np.argmax(h))
    if peak == 0:
        return 128
    x0, y0 = 0.0, float(h[0])
    x1, y1 = float(peak), float(h[peak])
    best_d, best_i = -1.0, 1
    for i in range(1, peak):
        xi, yi = float(i), float(h[i])
        d = abs((y1 - y0) * xi - (x1 - x0) * yi + x1 * y0 - y1 * x0) / max(1e-9, np.hypot(y1 - y0, x1 - x0))
        if d > best_d:
            best_d, best_i = d, i
    t = 255 - best_i if flip else best_i
    return int(np.clip(t, 1, 254))


def edge_map(gray: np.ndarray) -> np.ndarray:
    y_grad, x_grad = np.gradient(gray.astype(np.float64) / 255.0)
    mag = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    p95 = np.percentile(mag, 95)
    if p95 > 0:
        mag = mag / p95
    return np.clip(mag, 0.0, 1.0)


def density_map(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = gray.shape
    small_size = (max(1, w // scale), max(1, h // scale))
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    return np.array(image.resize(small_size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0


def fit_for_scoring(image: Image.Image, target_size: tuple[int, int], max_side: int) -> Image.Image:
    if max_side > 0:
        scale = min(max_side / max(target_size), 1.0)
        target_size = (max(1, round(target_size[0] * scale)), max(1, round(target_size[1] * scale)))
    return ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def downsample_for_variant_scoring(image: Image.Image, max_side: int) -> Image.Image:
    scoring = image.copy()
    side = max(128, max_side * 2)
    scoring.thumbnail((side, side), Image.Resampling.LANCZOS)
    return scoring


def crop_box_for_aspect(image: Image.Image, aspect: float, scale: float, offset_x: float, offset_y: float) -> tuple[int, int, int, int]:
    width, height = image.size
    image_aspect = width / height
    if image_aspect > aspect:
        crop_h = height
        crop_w = crop_h * aspect
    else:
        crop_w = width
        crop_h = crop_w / aspect
    crop_w *= scale
    crop_h *= scale
    crop_w = max(1.0, min(float(width), crop_w))
    crop_h = max(1.0, min(float(height), crop_h))
    slack_x = width - crop_w
    slack_y = height - crop_h
    left = slack_x * (0.5 + 0.5 * offset_x)
    top = slack_y * (0.5 + 0.5 * offset_y)
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + crop_w)),
        int(round(top + crop_h)),
    )


def median_fill_color(image: Image.Image) -> tuple[int, int, int]:
    arr = np.array(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
    med = np.median(arr.reshape(-1, 3), axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def apply_variant(image: Image.Image, spec: VariantSpec, aspect: float) -> Image.Image:
    work = image
    if spec.rotation:
        work = work.rotate(
            spec.rotation,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=median_fill_color(work),
        )
    box = crop_box_for_aspect(work, aspect, spec.scale, spec.offset_x, spec.offset_y)
    work = work.crop(box)
    if spec.brightness != 1.0:
        work = ImageEnhance.Brightness(work).enhance(spec.brightness)
    if spec.contrast != 1.0:
        work = ImageEnhance.Contrast(work).enhance(spec.contrast)
    if spec.local_contrast > 0:
        boosted = ImageOps.autocontrast(work).filter(
            ImageFilter.UnsharpMask(radius=1.4, percent=140, threshold=2)
        )
        work = Image.blend(work, boosted, spec.local_contrast)
    if spec.sharpness != 1.0:
        work = ImageEnhance.Sharpness(work).enhance(spec.sharpness)
    return work


def pre_score_variant(variant: Image.Image, target: Image.Image, target_binary: np.ndarray, max_side: int) -> float:
    scored = fit_for_scoring(variant, target.size, max_side)
    target_scored = fit_for_scoring(target, target.size, max_side)
    gray = rgb_to_gray(scored)
    target_gray = rgb_to_gray(target_scored)
    target_white_ratio = float(np.mean(target_binary == 255))
    target_binary_scored = np.where(target_gray >= otsu_threshold(target_gray), 255, 0).astype(np.uint8)

    def _score_for_binary(binary: np.ndarray) -> float:
        pixel_error = float(np.mean((binary / 255.0 - target_binary_scored / 255.0) ** 2))
        density_error = float(np.mean((density_map(gray) - density_map(target_gray)) ** 2))
        binary_density_error = float(np.mean((density_map(binary) - density_map(target_binary_scored)) ** 2))
        edge_error = float(np.mean(np.abs(edge_map(gray) - edge_map(target_gray))))
        ratio_error = abs(float(np.mean(binary == 255)) - target_white_ratio)
        return 0.34 * density_error + 0.24 * binary_density_error + 0.18 * edge_error + 0.14 * pixel_error + 0.10 * ratio_error

    t_q = int(np.clip(np.quantile(gray, 1.0 - target_white_ratio), 1, 254))
    bin_q = np.where(gray >= t_q, 255, 0).astype(np.uint8)
    t_o = otsu_threshold(gray)
    bin_o = np.where(gray >= t_o, 255, 0).astype(np.uint8)
    t_oc = otsu_threshold_clipped(gray)
    bin_oc = np.where(gray >= t_oc, 255, 0).astype(np.uint8)
    t_li = li_threshold(gray)
    bin_li = np.where(gray >= t_li, 255, 0).astype(np.uint8)
    t_tr = triangle_threshold(gray)
    bin_tr = np.where(gray >= t_tr, 255, 0).astype(np.uint8)
    gray_blur = np.array(
        Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float64,
    )
    t_ob = otsu_threshold(gray_blur)
    bin_ob = np.where(gray_blur >= t_ob, 255, 0).astype(np.uint8)

    parts = [
        _score_for_binary(bin_q),
        _score_for_binary(bin_o),
        _score_for_binary(bin_oc),
        _score_for_binary(bin_li),
        _score_for_binary(bin_tr),
        _score_for_binary(bin_ob),
    ]
    return float(np.mean(parts))


def build_specs(
    mode: str,
    center_scale: float = 0.97,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.35,
    center_brightness: float = 0.96,
    center_contrast: float = 0.96,
) -> list[VariantSpec]:
    specs = [
        VariantSpec(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, "baseline"),
    ]
    if mode == "local":
        scales = tuple(round(center_scale + delta, 3) for delta in (-0.015, -0.006, 0.0, 0.006, 0.015))
        offset_pairs = tuple(
            (round(center_offset_x + dx, 3), round(center_offset_y + dy, 3))
            for dx, dy in (
                (0.0, 0.0),
                (-0.08, 0.0),
                (0.08, 0.0),
                (0.0, -0.08),
                (0.0, 0.08),
            )
        )
        rotations = (0.0, -0.35, 0.35)
        contrasts = tuple(round(center_contrast + delta, 3) for delta in (-0.025, 0.0, 0.025))
        brightnesses = tuple(round(center_brightness + delta, 3) for delta in (-0.022, 0.0, 0.022))
        local_contrasts = (0.0, 0.22)
    elif mode == "dense":
        scales = tuple(round(s, 3) for s in (0.88, 0.91, 0.935, 0.955, 0.97, 0.985, 1.0))
        offset_pairs = tuple(
            (round(center_offset_x + dx, 3), round(center_offset_y + dy, 3))
            for dx, dy in (
                (0.0, 0.0),
                (-0.12, 0.0),
                (0.12, 0.0),
                (0.0, -0.12),
                (0.0, 0.12),
                (-0.08, -0.08),
                (0.08, 0.08),
                (-0.06, 0.18),
                (0.06, -0.18),
                (-0.2, 0.28),
                (0.2, -0.28),
            )
        )
        rotations = (-1.4, -0.9, -0.45, 0.0, 0.45, 0.9, 1.4)
        contrasts = tuple(round(center_contrast + d, 3) for d in (-0.06, -0.03, 0.0, 0.03, 0.06, 0.1))
        brightnesses = tuple(round(center_brightness + d, 3) for d in (-0.06, -0.03, 0.0, 0.03, 0.06))
        local_contrasts = (0.0, 0.18, 0.35, 0.5)
    elif mode == "explore":
        scales = (0.89, 0.97, 1.0)
        offset_pairs = (
            (0.0, 0.0),
            (-0.42, 0.0),
            (0.42, 0.0),
            (0.0, -0.42),
            (0.0, 0.42),
            (-0.28, -0.28),
            (0.28, 0.28),
            (-0.18, 0.38),
            (0.18, -0.38),
        )
        rotations = (-1.4, -0.6, 0.0, 0.6, 1.4)
        contrasts = (0.9, 0.96, 1.04, 1.12)
        brightnesses = (0.9, 0.96, 1.04, 1.1)
        local_contrasts = (0.0, 0.22, 0.42)
    elif mode == "quick":
        scales = (1.0, 0.96, 0.92)
        offset_pairs = ((0.0, 0.0), (-0.35, 0.0), (0.35, 0.0))
        rotations = (0.0, -0.8, 0.8)
        contrasts = (1.0, 1.12)
        brightnesses = (1.0, 1.04)
        local_contrasts = (0.0, 0.35)
    else:
        scales = (1.0, 0.98, 0.95, 0.92, 0.89)
        offset_pairs = (
            (0.0, 0.0),
            (-0.35, 0.0),
            (0.35, 0.0),
            (0.0, -0.35),
            (0.0, 0.35),
            (-0.25, -0.25),
            (0.25, 0.25),
            (-0.12, 0.42),
            (0.12, -0.42),
        )
        rotations = (0.0, -1.2, -0.6, 0.6, 1.2)
        contrasts = (0.92, 0.96, 1.06, 1.12)
        brightnesses = (0.92, 0.96, 1.04, 1.08)
        local_contrasts = (0.0, 0.28, 0.42)
    if mode == "explore":
        sharp_loop = (0.98, 1.06, 1.14)
    elif mode == "aggressive":
        sharp_loop = (1.0, 1.12)
    else:
        sharp_loop = (None,)

    for scale in scales:
        for offset_x, offset_y in offset_pairs:
            for rotation in rotations:
                for contrast in contrasts:
                    for brightness in brightnesses:
                        for local_contrast in local_contrasts:
                            for shp in sharp_loop:
                                sharp_val = shp if shp is not None else (1.12 if local_contrast else 1.0)
                                specs.append(
                                    VariantSpec(
                                        scale,
                                        offset_x,
                                        offset_y,
                                        rotation,
                                        brightness,
                                        contrast,
                                        local_contrast,
                                        sharp_val,
                                        "grid",
                                    )
                                )
    unique: list[VariantSpec] = []
    seen: set[VariantSpec] = set()
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        unique.append(spec)
    return unique


def prepare_output_dir(out_dir: Path, resume: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    for pattern in ("variant_*.png", "variants.sqlite", "variants_manifest.jsonl", "index.html"):
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
        CREATE TABLE IF NOT EXISTS variants (
            id INTEGER PRIMARY KEY,
            output_file TEXT NOT NULL,
            pre_score REAL NOT NULL,
            scale REAL NOT NULL,
            offset_x REAL NOT NULL,
            offset_y REAL NOT NULL,
            rotation REAL NOT NULL,
            brightness REAL NOT NULL,
            contrast REAL NOT NULL,
            local_contrast REAL NOT NULL,
            sharpness REAL NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_variants_pre_score ON variants(pre_score)")
    conn.commit()
    return conn


def make_thumbnail(src: Path, dst: Path, side: int = 180) -> None:
    with Image.open(src) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((side, side), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (side, side), (15, 23, 42))
        canvas.paste(thumb, ((side - thumb.width) // 2, (side - thumb.height) // 2))
        canvas.save(dst, optimize=True)


def write_html(out_dir: Path, rows: list[VariantResult]) -> None:
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    for row in rows:
        make_thumbnail(out_dir / row.output_file, thumbs / row.output_file)
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card">
              <a href="{escape(row.output_file)}" target="_blank"><img src="thumbs/{escape(row.output_file)}" alt="variant {row.id}"></a>
              <div class="meta">
                <strong>#{row.id:04d} {escape(row.source)}</strong>
                <span>pre-score <b>{row.pre_score:.5f}</b></span>
                <span>scale {row.scale:.2f}</span><span>rot {row.rotation:+.1f}</span>
                <span>off {row.offset_x:+.2f},{row.offset_y:+.2f}</span>
                <span>b {row.brightness:.2f}</span><span>c {row.contrast:.2f}</span>
                <span>local {row.local_contrast:.2f}</span>
              </div>
            </article>
            """
        )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Variantes de entrada laser</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070b12; --panel:#0f172a; --muted:#8b9cb3; --text:#e5edf7; --accent:#38bdf8; --border:rgba(148,163,184,.22); }}
    body {{ margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:20px 24px; border-bottom:1px solid var(--border); background:#08111f; }}
    h1 {{ margin:0 0 8px; }}
    p {{ margin:0; color:var(--muted); }}
    main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px; padding:18px; }}
    .card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
    .card img {{ width:100%; display:block; background:#020617; }}
    .meta {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 8px; padding:10px; font-size:.82rem; color:var(--muted); }}
    .meta strong {{ grid-column:1/-1; color:var(--text); }}
    b {{ color:var(--text); }}
    a {{ color:#dff7ff; }}
  </style>
</head>
<body>
  <header>
    <h1>Variantes de entrada laser</h1>
    <p>{len(rows)} variantes pre-rankeadas. Usar esta carpeta como <code>--input-dir</code> en <code>laser_batch_match.py</code>.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    out_dir.joinpath("index.html").write_text(html, encoding="utf-8")


def select_variants(scored: list[tuple[float, VariantSpec]], limit: int, keep_per_bucket: int) -> list[tuple[float, VariantSpec]]:
    selected: list[tuple[float, VariantSpec]] = []
    seen: set[VariantSpec] = set()
    for item in sorted(scored, key=lambda row: row[0]):
        if item[1].source == "baseline":
            selected.append(item)
            seen.add(item[1])
            break
    buckets: dict[tuple[float, float], int] = {}
    for score, spec in sorted(scored, key=lambda row: row[0]):
        if spec in seen:
            continue
        bucket = (spec.scale, spec.rotation)
        if buckets.get(bucket, 0) >= keep_per_bucket:
            continue
        selected.append((score, spec))
        seen.add(spec)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if len(selected) >= limit:
            break
    for item in sorted(scored, key=lambda row: row[0]):
        if len(selected) >= limit:
            break
        if item[1] in seen:
            continue
        selected.append(item)
        seen.add(item[1])
    return sorted(selected, key=lambda row: row[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera variantes de entrada pre-rankeadas contra un target")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=120, help="Cuantas variantes guardar para el batch (sube a 100–200+ con --mode explore)")
    parser.add_argument("--mode", choices=("quick", "aggressive", "local", "dense", "explore"), default="aggressive")
    parser.add_argument("--score-max-side", type=int, default=260)
    parser.add_argument(
        "--keep-per-bucket",
        type=int,
        default=5,
        help="Máx. variantes por bucket (scale, rotación) antes de rellenar hasta --limit",
    )
    parser.add_argument("--center-scale", type=float, default=0.97)
    parser.add_argument("--center-offset-x", type=float, default=0.0)
    parser.add_argument("--center-offset-y", type=float, default=0.35)
    parser.add_argument("--center-brightness", type=float, default=0.96)
    parser.add_argument("--center-contrast", type=float, default=0.96)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_image = load_rgb(args.input)
    scoring_input = downsample_for_variant_scoring(input_image, args.score_max_side)
    target_image = load_rgb(args.target)
    aspect = target_image.width / target_image.height
    target_gray = rgb_to_gray(fit_for_scoring(target_image, target_image.size, args.score_max_side))
    target_binary = np.where(target_gray >= otsu_threshold(target_gray), 255, 0).astype(np.uint8)

    prepare_output_dir(args.out, args.resume)
    scored: list[tuple[float, VariantSpec]] = []
    specs = build_specs(
        args.mode,
        args.center_scale,
        args.center_offset_x,
        args.center_offset_y,
        args.center_brightness,
        args.center_contrast,
    )
    for index, spec in enumerate(specs, start=1):
        variant = apply_variant(scoring_input, spec, aspect)
        score = pre_score_variant(variant, target_image, target_binary, args.score_max_side)
        scored.append((score, spec))
        if index % 500 == 0:
            best = min(scored, key=lambda row: row[0])
            print(f"{index}/{len(specs)} pre-score best={best[0]:.5f}", flush=True)

    selected = select_variants(scored, args.limit, args.keep_per_bucket)
    conn = ensure_db(args.out / "variants.sqlite")
    rows: list[VariantResult] = []
    manifest = args.out / "variants_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row_id, (score, spec) in enumerate(selected, start=1):
            variant = apply_variant(input_image, spec, aspect)
            output_file = f"variant_{row_id:04d}.png"
            variant.save(args.out / output_file, optimize=True)
            result = VariantResult(row_id, output_file, score, **asdict(spec))
            rows.append(result)
            payload = asdict(result)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            conn.execute(
                """
                INSERT OR REPLACE INTO variants (
                    id, output_file, pre_score, scale, offset_x, offset_y, rotation,
                    brightness, contrast, local_contrast, sharpness, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.output_file,
                    result.pre_score,
                    result.scale,
                    result.offset_x,
                    result.offset_y,
                    result.rotation,
                    result.brightness,
                    result.contrast,
                    result.local_contrast,
                    result.sharpness,
                    result.source,
                ),
            )
    conn.commit()
    conn.close()
    write_html(args.out, rows)

    best = rows[0]
    print(f"Listo: {args.out}")
    print(f"  Variantes generadas: {len(rows)} de {len(specs)} evaluadas")
    print(f"  SQLite: {args.out / 'variants.sqlite'}")
    print(f"  Reporte: {args.out / 'index.html'}")
    print(f"  Mejor pre-score: {best.output_file} score={best.pre_score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
