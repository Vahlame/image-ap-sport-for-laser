#!/usr/bin/env python3
"""
Replica un candidato guardado en match.sqlite y desglosa el score (v2, v3 o v4).

Guarda:
  - match_row.json: fila SQLite
  - 00..04: etapas de preprocess_gray (autocontrast, gamma+contraste+brillo, sharpen, invert)
  - 05_render_output.png: salida 1-bit del algoritmo elegido
  - score_breakdown.json: terminos numericos (v2/v3/v4)
  - explain.txt: resumen en texto

Requiere el mismo preprocess que la corrida original (este script solo soporta --preprocess-mode none
implicito: no aplica sauvola/niblack; usa la misma ruta que campanas `preprocess-mode none`).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import laser_scoring as ls
from laser_target_match import (
    candidate_from_row,
    detect_binary_target,
    load_rgb,
    otsu_threshold,
    render_candidate,
    resize_to_target,
    rgb_to_gray,
    set_gray_luma_standard,
)


def preprocess_gray_stages(base_gray: np.ndarray, candidate: object) -> list[tuple[str, np.ndarray]]:
    """Misma secuencia que preprocess_gray en laser_target_match (para PNGs explicativos)."""
    stages: list[tuple[str, np.ndarray]] = []
    gray = base_gray.copy()
    stages.append(("00_input_gray", np.clip(gray, 0, 255).astype(np.uint8)))

    if candidate.autocontrast > 0:
        low = np.percentile(gray, candidate.autocontrast)
        high = np.percentile(gray, 100.0 - candidate.autocontrast)
        if high > low:
            gray = (gray - low) * (255.0 / (high - low))
    gray = np.clip(gray, 0.0, 255.0)
    stages.append(("01_after_autocontrast_clip", gray.astype(np.uint8)))

    if candidate.gamma != 1.0:
        gray = 255.0 * np.power(gray / 255.0, 1.0 / candidate.gamma)
    gray = (gray - 128.0) * candidate.contrast + 128.0 + candidate.brightness
    gray = np.clip(gray, 0.0, 255.0)
    stages.append(("02_after_gamma_contrast_brightness", gray.astype(np.uint8)))

    if candidate.sharpen > 0:
        image = Image.fromarray(gray.astype(np.uint8), mode="L")
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(candidate.sharpen), threshold=2))
        gray = np.array(image, dtype=np.float64)
    stages.append(("03_after_sharpen", np.clip(gray, 0, 255).astype(np.uint8)))

    if candidate.invert:
        gray = 255.0 - gray
    stages.append(("04_after_invert_ready_for_dither", np.clip(gray, 0, 255).astype(np.uint8)))
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description="Explica score v2/v3/v4 y reproduce raster desde match.sqlite")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--sqlite", type=Path, required=True)
    ap.add_argument("--match-id", type=int, default=0, help="0 = mejor fila ORDER BY score ASC")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--luma", choices=("bt601", "bt709"), default="bt601")
    ap.add_argument(
        "--score-version",
        choices=("v2", "v3", "v4"),
        default="v2",
        help="Metrica para el desglose (la fila SQLite puede haberse evaluado con otra version).",
    )
    args = ap.parse_args()

    if not args.sqlite.is_file():
        print(f"No existe SQLite: {args.sqlite}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    set_gray_luma_standard(args.luma)

    input_image = load_rgb(args.input)
    target_image_full = load_rgb(args.target)
    target_image = target_image_full.copy()
    if args.max_side > 0:
        target_image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
    input_resized = resize_to_target(input_image, target_image.size, max_side=0)
    rgb_u8 = np.array(input_resized)
    base_gray = rgb_to_gray(rgb_u8)

    target_gray = rgb_to_gray(np.array(target_image.convert("RGB")))
    if detect_binary_target(target_gray):
        target_binary = np.where(target_gray > 127, 255, 0).astype(np.uint8)
    else:
        tt = otsu_threshold(target_gray)
        target_binary = np.where(target_gray >= tt, 255, 0).astype(np.uint8)
    Image.fromarray(target_binary, mode="L").save(args.out / "target_binary.png", optimize=True)
    target_density = ls.density_map(target_gray)
    target_edges = ls.edge_map(target_gray)

    with sqlite3.connect(args.sqlite) as conn:
        conn.row_factory = sqlite3.Row
        if args.match_id > 0:
            row = conn.execute("SELECT * FROM matches WHERE id = ?", (args.match_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM matches ORDER BY score ASC LIMIT 1").fetchone()
    if row is None:
        print("Sin filas en matches", file=sys.stderr)
        return 2

    cand = candidate_from_row(row)
    meta = {k: row[k] for k in row.keys()}
    (args.out / "match_row.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name, arr in preprocess_gray_stages(base_gray, cand):
        Image.fromarray(arr, mode="L").save(args.out / f"{name}.png", optimize=True)

    out = render_candidate(base_gray, cand)
    Image.fromarray(out, mode="L").save(args.out / "05_render_output.png", optimize=True)

    if args.score_version == "v3":
        terms = ls.score_candidate_v3_terms(out, target_gray, target_binary, target_density, target_edges, cand)
        doc = (
            f"Score v3 (replicado, score_version={args.score_version})\n"
            "==================================================\n\n"
            "v3 = 0.28*(1-SSIM_cont) + 0.22*(1-SSIM_blur_bin) + 0.18*MSE_bin + 0.14*edge "
            "+ 0.09*density + 0.04*ratio + 0.05*reg\n\n"
            "SSIM_blur_bin: SSIM entre salida y target binario tras Gaussian sigma~1.15 "
            "(halftone vs mascara, ver ssim_blur_* en JSON).\n\n"
            "Ver score_breakdown.json.\n"
        )
    elif args.score_version == "v4":
        terms = ls.score_candidate_v4_terms(out, target_gray, target_binary, target_density, target_edges, cand)
        doc = (
            f"Score v4 (replicado, score_version={args.score_version})\n"
            "=================================================\n\n"
            "v4 = 0.24*(1-SSIM_blur_sym) + 0.12*MSE_blur + 0.20*LPIPS_scaled(blur) "
            "+ 0.15*MSE_bin + 0.12*edge + 0.07*density + 0.05*ratio + 0.05*reg\n\n"
            "Sin SSIM continuo directo binario-vs-gris. Blur gaussiano simetrico (blur_sigma en JSON); "
            "LPIPS Alex sobre gris triplicado (requiere torch+lpips).\n\n"
            "Ver score_breakdown.json.\n"
        )
    else:
        terms = ls.score_candidate_v2_terms(out, target_gray, target_binary, target_density, target_edges, cand)
        doc = (
            f"Score v2 (replicado, score_version={args.score_version})\n"
            "====================================\n\n"
            "score = 0.40*(1-SSIM) + 0.20*MSE_bin + 0.15*edge_err + 0.10*density_err "
            "+ 0.05*ratio_err + 0.10*reg\n\n"
            "- SSIM (ssim_raw): salida 0..1 vs target GRIS continuo (no vs binario).\n"
            "- MSE_bin (pixel_error): salida vs target_binary.\n"
            "- edge_err / density_err: vs mapas del target gris.\n"
            "- ratio_err: |fraccion blancos salida - target|.\n"
            "- reg: penalizacion por contraste bajo, brillo extremo o sharpen alto.\n\n"
            "Ver score_breakdown.json.\n"
        )
    terms_out = dict(terms)
    terms_out["explain_score_version"] = args.score_version
    (args.out / "score_breakdown.json").write_text(json.dumps(terms_out, indent=2), encoding="utf-8")
    lines = [doc, "\nCampos (clave = valor):\n"]
    for k in sorted(terms.keys()):
        lines.append(f"  {k} = {terms[k]:.10g}\n")
    (args.out / "explain.txt").write_text("".join(lines), encoding="utf-8")

    print(json.dumps(terms, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
