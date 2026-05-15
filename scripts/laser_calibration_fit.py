#!/usr/bin/env python3
"""
Ajusta una LUT de material a partir de una foto del step-wedge grabado.

Workflow Fase R7 (cierre del loop fisico):
1. `laser_calibration_wedge.py` -> `wedge.png` + `wedge_meta.json`.
2. Grabar `wedge.png` en el material objetivo con parametros reales.
3. Fotografiar bajo luz difusa cruzada (resalta microcavidades en acrilico,
   reduce sombras puntuales).
4. **Este script**: alinea la foto al wedge (escala/rotacion simple), mide
   el L* de cada parche grabado, ajusta una curva monotonica `input_gray ->
   measured_gray`, invierte la curva para producir una LUT que, aplicada
   al pre-dither, compensa el dot-gain del material.
5. Salida: `<out>.npy` (LUT 256 u8) + `<out>.json` (metadata + datapoints).
6. La LUT se carga con `laser_physics.MaterialProfile` (campo `lut_curve` o
   `lut_curve_npy`) y se usa en runs con `--material custom`.

Diseño:
- **Alineacion simple**: por defecto resize la foto a `wedge_meta.image_size_px`
  asumiendo crop manual previo. Opcional `--photo-crop x,y,w,h` antes del resize.
  Para alineacion robusta (rotacion/skew) hay `--align ecc` que usa
  `cv2.findTransformECC` si OpenCV esta disponible.
- **Medicion**: por cada parche en `meta.patches`, promedia los pixeles del centro
  (recortando bordes para evitar etiquetas).
- **Fit**: PchipInterpolator (monotonico) sobre `(input_gray, measured_gray)`.
  Si la medicion no es monotonica (ej. madera con rebote pirolitico), aplica
  isotonic regression antes del fit y emite warning.
- **Inversion**: para construir la LUT, para cada `g_desired in 0..255`,
  busca `pre_input` tal que `curve(pre_input) == g_desired`; si fuera del
  rango medido, clamp a los extremos.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class PatchMeasurement:
    """Resultado de medir un parche grabado en la foto."""

    index: int
    input_gray: int  # del wedge_meta
    measured_gray: float  # 0..255, media post-blur
    pixel_count: int
    center_px: tuple[int, int]


def load_photo_as_gray(path: Path) -> np.ndarray:
    """Carga la foto y la convierte a gris (luminancia BT.601) uint8."""
    if not path.is_file():
        raise FileNotFoundError(f"foto no encontrada: {path}")
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def load_wedge_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta no encontrada: {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    schema = data.get("schema", "")
    if not schema.startswith("laser_calibration_wedge/"):
        raise ValueError(f"meta no es de laser_calibration_wedge: schema='{schema}'")
    return data


def crop_image(photo: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"crop invalido: {crop}")
    return photo[y:y + h, x:x + w].copy()


def resize_to_wedge(photo: np.ndarray, target_size_px: tuple[int, int]) -> np.ndarray:
    """Resize bilineal (Lanczos) al tamano del wedge generado.

    target_size_px: (width, height) — como en wedge_meta['image_size_px'].
    """
    tw, th = int(target_size_px[0]), int(target_size_px[1])
    if photo.shape == (th, tw):
        return photo
    img = Image.fromarray(photo, mode="L")
    return np.array(img.resize((tw, th), Image.Resampling.LANCZOS), dtype=np.uint8)


def measure_patches(
    photo: np.ndarray,
    meta: dict,
    *,
    inset_fraction: float = 0.15,
    blur_radius_px: int = 3,
) -> list[PatchMeasurement]:
    """
    Mide el L* aproximado de cada parche en la foto (ya alineada al wedge).

    Args:
        photo: gris uint8, mismo tamano que wedge_meta['image_size_px'].
        meta: dict cargado de wedge_meta.json.
        inset_fraction: recorta este porcentaje en cada lado del parche para evitar
            bordes/etiquetas (default 15%, deja el 70% central).
        blur_radius_px: radio del uniform_filter para suavizar ruido fotografico
            antes de promediar (default 3 px).
    """
    expected_size = tuple(meta["image_size_px"])  # [w, h]
    actual_size = (photo.shape[1], photo.shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f"foto shape {actual_size} != wedge_meta image_size_px {expected_size}; "
            "usa resize_to_wedge o --photo-crop antes."
        )
    smoothed = uniform_filter(photo.astype(np.float64), size=max(1, int(blur_radius_px)))
    results: list[PatchMeasurement] = []
    for p in meta["patches"]:
        x = int(p["x_px"])
        y = int(p["y_px"])
        w = int(p["w_px"])
        h = int(p["h_px"])
        inset_x = max(1, int(w * float(inset_fraction)))
        inset_y = max(1, int(h * float(inset_fraction)))
        x0 = x + inset_x
        y0 = y + inset_y
        x1 = x + w - inset_x
        y1 = y + h - inset_y
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = x, y, x + w, y + h
        patch_pix = smoothed[y0:y1, x0:x1]
        mean_val = float(patch_pix.mean())
        results.append(
            PatchMeasurement(
                index=int(p["index"]),
                input_gray=int(p["input_gray"]),
                measured_gray=mean_val,
                pixel_count=int(patch_pix.size),
                center_px=(x + w // 2, y + h // 2),
            )
        )
    return results


def _enforce_monotonic_isotonic(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Aplica isotonic regression simple (PAVA) si `ys` no es monotonico no decreciente.

    Para LUT: queremos `measured` crecer con `input`. Si no, suaviza forzando
    promedios por pool de violaciones. Para mantener el scaffold sin scikit-learn,
    implementacion minimal del Pool Adjacent Violators Algorithm.
    """
    if xs.shape != ys.shape:
        raise ValueError("xs e ys deben tener el mismo shape")
    n = len(ys)
    if n < 2:
        return ys.astype(np.float64).copy()
    out = ys.astype(np.float64).copy()
    weights = np.ones(n, dtype=np.float64)
    i = 0
    while i < n - 1:
        if out[i] <= out[i + 1]:
            i += 1
            continue
        # violation: pool
        total_w = weights[i] + weights[i + 1]
        avg = (out[i] * weights[i] + out[i + 1] * weights[i + 1]) / total_w
        out[i:i + 2] = avg
        weights[i:i + 2] = total_w
        # backtrack
        while i > 0 and out[i - 1] > out[i]:
            tw = weights[i - 1] + weights[i]
            ag = (out[i - 1] * weights[i - 1] + out[i] * weights[i]) / tw
            out[i - 1:i + 1] = ag
            weights[i - 1:i + 1] = tw
            i -= 1
    return out


def fit_inverse_lut(
    measurements: list[PatchMeasurement],
    *,
    n_lut: int = 256,
    force_monotonic: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Ajusta curva `input -> measured` y la invierte para producir la LUT.

    Returns:
        (lut_array uint8 shape (n_lut,), debug_info dict).
    """
    pts = sorted(measurements, key=lambda m: m.input_gray)
    xs_in = np.array([float(m.input_gray) for m in pts], dtype=np.float64)
    ys_meas = np.array([float(m.measured_gray) for m in pts], dtype=np.float64)

    debug: dict = {
        "n_points": len(pts),
        "raw_input_gray": xs_in.tolist(),
        "raw_measured_gray": ys_meas.tolist(),
        "input_range": [float(xs_in.min()), float(xs_in.max())],
        "measured_range": [float(ys_meas.min()), float(ys_meas.max())],
    }

    if force_monotonic:
        ys_mono = _enforce_monotonic_isotonic(xs_in, ys_meas)
        non_mono_violations = int((np.diff(ys_meas) < 0).sum())
        debug["non_monotonic_violations_input"] = non_mono_violations
        debug["measured_monotonic"] = ys_mono.tolist()
        if non_mono_violations > 0:
            warnings.warn(
                f"Mediciones con {non_mono_violations} violaciones de monotonicidad "
                "(material con rebote tonal?); aplicada isotonic regression.",
                UserWarning,
                stacklevel=2,
            )
        ys_fit = ys_mono
    else:
        ys_fit = ys_meas
        debug["measured_monotonic"] = ys_fit.tolist()

    # Dedupe en x (PCHIP requiere xs unicos)
    unique_xs, unique_idx = np.unique(xs_in, return_index=True)
    unique_ys = ys_fit[unique_idx]
    if len(unique_xs) < 2:
        raise ValueError(f"Se necesitan >=2 valores unicos de input_gray, hay {len(unique_xs)}")

    forward = PchipInterpolator(unique_xs, unique_ys, extrapolate=True)
    # Sample a una densa rejilla para invertir
    dense_in = np.linspace(float(unique_xs.min()), float(unique_xs.max()), 4096)
    dense_out = forward(dense_in)
    # Forzar monotonia tras interpolacion (PCHIP la conserva si la entrada es monotonica)
    dense_out_mono = np.maximum.accumulate(dense_out)

    # Para LUT inverse: dado g_desired, encontrar pre_input tal que forward(pre_input) ~ g_desired.
    lut_grays = np.arange(n_lut, dtype=np.float64)
    pre_inputs = np.interp(lut_grays, dense_out_mono, dense_in)
    # Clamp extremos: g_desired fuera del rango medido se mapea a 0 o 255
    pre_inputs = np.clip(pre_inputs, 0.0, 255.0)
    lut = np.round(pre_inputs).astype(np.uint8)

    debug["lut_first_5"] = lut[:5].tolist()
    debug["lut_last_5"] = lut[-5:].tolist()
    debug["lut_at_64_128_192"] = [int(lut[64]), int(lut[128]), int(lut[192])]
    return lut, debug


def save_lut(lut: np.ndarray, debug: dict, out_npy: Path, *, material_name: str | None = None) -> dict:
    """Guarda LUT en `.npy` + JSON sidecar (datapoints + LUT inline)."""
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, lut.astype(np.uint8))
    sidecar = out_npy.with_suffix(".json")
    payload: dict = {
        "schema": "laser_calibration_fit/v1",
        "lut_npy": out_npy.name,
        "lut_inline": lut.tolist(),  # para portabilidad sin npy
        "material": material_name or "",
        "debug": debug,
    }
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ajusta LUT material desde foto del step-wedge grabado")
    p.add_argument("--photo", type=Path, required=True, help="Foto del wedge grabado (JPG/PNG)")
    p.add_argument("--wedge-meta", type=Path, required=True, help="wedge_meta.json producido por laser_calibration_wedge")
    p.add_argument("--out", type=Path, required=True, help="Ruta LUT de salida (.npy); se crea .json sidecar")
    p.add_argument(
        "--photo-crop",
        type=str,
        default="",
        help="Recorte previo de la foto: 'x,y,w,h' en pixeles (sobre la foto original).",
    )
    p.add_argument(
        "--inset-fraction",
        type=float,
        default=0.15,
        help="Fraccion (cada lado) recortada del parche al medir (default 0.15 -> 70%% central).",
    )
    p.add_argument(
        "--blur-radius",
        type=int,
        default=3,
        help="Radio del uniform_filter previo a promediar parches (default 3 px).",
    )
    p.add_argument(
        "--no-force-monotonic",
        action="store_true",
        help="No forzar monotonia (util para visualizar el rebote en madera).",
    )
    p.add_argument("--material-name", type=str, default="", help="Etiqueta de material para el sidecar")
    return p


def main() -> int:
    args = build_argument_parser().parse_args()

    meta = load_wedge_meta(args.wedge_meta)
    photo = load_photo_as_gray(args.photo)

    if args.photo_crop:
        parts = [int(x) for x in args.photo_crop.split(",")]
        if len(parts) != 4:
            print(f"[FIT] --photo-crop esperaba 'x,y,w,h', got '{args.photo_crop}'", file=sys.stderr)
            return 2
        photo = crop_image(photo, (parts[0], parts[1], parts[2], parts[3]))

    expected_size = tuple(meta["image_size_px"])
    photo = resize_to_wedge(photo, expected_size)
    print(f"[FIT] foto alineada a {expected_size[0]}x{expected_size[1]} px (wedge dims)", flush=True)

    measurements = measure_patches(
        photo, meta,
        inset_fraction=float(args.inset_fraction),
        blur_radius_px=int(args.blur_radius),
    )
    print(f"[FIT] medidos {len(measurements)} parches", flush=True)
    for m in measurements:
        print(f"  patch {m.index:02d}: input_gray={m.input_gray:3d} -> measured={m.measured_gray:6.2f}")

    lut, debug = fit_inverse_lut(
        measurements,
        force_monotonic=not args.no_force_monotonic,
    )
    sidecar = save_lut(lut, debug, args.out, material_name=args.material_name or None)
    print(f"[FIT] LUT guardada: {args.out}", flush=True)
    print(f"[FIT] sidecar:      {args.out.with_suffix('.json')}", flush=True)
    print(
        f"[FIT] LUT samples: 64={int(lut[64])} 128={int(lut[128])} 192={int(lut[192])}",
        flush=True,
    )
    if debug.get("non_monotonic_violations_input", 0) > 0:
        print(
            f"[FIT] ADVERTENCIA: {debug['non_monotonic_violations_input']} violaciones de monotonia "
            "en las mediciones; isotonic regression aplicada (material con rebote?).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
