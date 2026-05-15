#!/usr/bin/env python3
"""
Genera tiras de calibracion (step-wedge) listas para grabar en la maquina real.

Uso del workflow Fase R7:
1. Generar el PNG: `python scripts/laser_calibration_wedge.py --out wedge.png --material acrylic_back_engrave`
2. Importar en LightBurn (o el CAM) con interval = 25.4/DPI mm; Pass-Through si aplica.
3. Grabar en el material objetivo con los parametros que vayas a usar (potencia/velocidad).
4. Fotografiar la tira bajo luz difusa cruzada (resalta microcavidades en acrilico).
5. (futuro) `laser_calibration_fit.py --photo foto.jpg --wedge-meta wedge_meta.json --out lut.npy`
   -> ajusta spline monotonica inversa input->output -> LUT calibrada por material*maquina.

El generador es agnostico de material; permite varias variantes:
- Step wedge tonal: N parches con grises 0..255, cada uno dithereado con el algoritmo elegido.
- Power-speed grid: matriz de M power% x N speed (placeholders solidos; metadata para que
  el CAM sepa que ajustar; el wedge tonal ya tiene los ranges en el material).
- Etiquetas opcionales con el valor de gris de cada parche (legibles a baja DPI).

Salidas:
- wedge.png: bitmap 1-bit listo para Pass-Through.
- wedge_meta.json: posiciones, valores, dither, DPI; consumido por el fit posterior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class PatchSpec:
    """Especificacion de un parche del step-wedge."""

    index: int
    input_gray: int  # 0..255
    x_px: int
    y_px: int
    w_px: int
    h_px: int
    dither: str
    label: str

    def center_px(self) -> tuple[int, int]:
        return (self.x_px + self.w_px // 2, self.y_px + self.h_px // 2)


def _mm_to_px(mm: float, dpi: int) -> int:
    return max(1, int(round(mm * dpi / 25.4)))


def _try_default_font(size_px: int) -> ImageFont.ImageFont | None:
    """Intenta cargar una fuente del sistema; cae a PIL default si no."""
    candidates = (
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    for p in candidates:
        try:
            return ImageFont.truetype(p, size_px)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default()
    except (OSError, IOError):
        return None


def step_gray_values(steps: int, *, gamma: float = 1.0) -> list[int]:
    """
    Genera N valores de gris para el step-wedge.

    `gamma`=1.0 da espaciado lineal en (0, 255). `gamma`>1 concentra valores en oscuros
    (util para acrilico back-engrave donde el rango util es 0..120 — querés mas resolucion
    en oscuros donde el dot-gain importa). `gamma`<1 concentra en claros.

    Implementacion: `raw ** gamma` (no `raw ** (1/gamma)`). Verificable:
      gamma=2.0 sobre 0.5 = 0.25 -> mas oscuro -> valores intermedios menores que lineal.
    """
    if steps < 2:
        raise ValueError(f"steps debe ser >= 2, got {steps}")
    raw = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    if gamma != 1.0:
        raw = np.power(raw, float(gamma))
    return [int(round(v * 255.0)) for v in raw]


def layout_patches(
    steps: int,
    *,
    square_mm: float,
    gap_mm: float,
    margin_mm: float,
    dpi: int,
    cols: int | None = None,
    dither: str = "floyd",
    gamma: float = 1.0,
) -> tuple[list[PatchSpec], int, int]:
    """
    Distribuye N parches en grilla. Devuelve specs, ancho total px, alto total px.

    Si `cols` es None: elige la grilla mas cuadrada posible (sqrt).
    """
    if steps < 2:
        raise ValueError(f"steps debe ser >= 2, got {steps}")
    if cols is None:
        cols = max(1, int(round(np.sqrt(steps))))
    rows = (steps + cols - 1) // cols
    sq_px = _mm_to_px(square_mm, dpi)
    gap_px = _mm_to_px(gap_mm, dpi)
    margin_px = _mm_to_px(margin_mm, dpi)

    total_w = margin_px * 2 + cols * sq_px + (cols - 1) * gap_px
    total_h = margin_px * 2 + rows * sq_px + (rows - 1) * gap_px

    grays = step_gray_values(steps, gamma=gamma)
    patches: list[PatchSpec] = []
    for i, g in enumerate(grays):
        r, c = divmod(i, cols)
        x = margin_px + c * (sq_px + gap_px)
        y = margin_px + r * (sq_px + gap_px)
        patches.append(
            PatchSpec(
                index=i,
                input_gray=int(g),
                x_px=int(x),
                y_px=int(y),
                w_px=int(sq_px),
                h_px=int(sq_px),
                dither=dither,
                label=str(int(g)),
            )
        )
    return patches, total_w, total_h


def _dither_floyd(gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Floyd-Steinberg minimal — sin importar laser_target_match (evita ciclos)."""
    work = gray.astype(np.float64).copy()
    h, w = work.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            old = work[y, x]
            new_v = 255.0 if old >= threshold else 0.0
            out[y, x] = 255 if new_v > 0 else 0
            err = old - new_v
            if x + 1 < w:
                work[y, x + 1] += err * 7.0 / 16.0
            if y + 1 < h:
                if x - 1 >= 0:
                    work[y + 1, x - 1] += err * 3.0 / 16.0
                work[y + 1, x] += err * 5.0 / 16.0
                if x + 1 < w:
                    work[y + 1, x + 1] += err * 1.0 / 16.0
    return out


def _dither_ordered_bayer8(gray: np.ndarray) -> np.ndarray:
    """Bayer 8x8 ordered dither, deterministico y rapido."""
    bayer = (1 / 65.0) * np.array(
        [
            [0, 48, 12, 60, 3, 51, 15, 63],
            [32, 16, 44, 28, 35, 19, 47, 31],
            [8, 56, 4, 52, 11, 59, 7, 55],
            [40, 24, 36, 20, 43, 27, 39, 23],
            [2, 50, 14, 62, 1, 49, 13, 61],
            [34, 18, 46, 30, 33, 17, 45, 29],
            [10, 58, 6, 54, 9, 57, 5, 53],
            [42, 26, 38, 22, 41, 25, 37, 21],
        ],
        dtype=np.float64,
    )
    h, w = gray.shape
    tile = np.tile(bayer, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return np.where(gray.astype(np.float64) / 255.0 >= tile, 255, 0).astype(np.uint8)


def _dither_blue_noise_vac32(gray: np.ndarray) -> np.ndarray:
    """Blue-noise via void-and-cluster (Ulichney) 32x32 cached."""
    try:
        import laser_blue_noise
    except ImportError as exc:
        raise RuntimeError("Para dither blue_noise_vac32 instala scipy y verifica laser_blue_noise.py") from exc
    t = laser_blue_noise.threshold_matrix_for_dithering(size=32)
    h, w = gray.shape
    tile = np.tile(t, (h // 32 + 1, w // 32 + 1))[:h, :w]
    return np.where(gray.astype(np.float64) / 255.0 >= tile, 255, 0).astype(np.uint8)


_DITHERS = {
    "floyd": _dither_floyd,
    "bayer8": _dither_ordered_bayer8,
    "blue_noise_vac32": _dither_blue_noise_vac32,
    "threshold": lambda g: np.where(g >= 128, 255, 0).astype(np.uint8),
}


def render_patch(patch: PatchSpec) -> np.ndarray:
    """Renderiza un parche: gris constante -> dither -> binario 0/255."""
    gray = np.full((patch.h_px, patch.w_px), patch.input_gray, dtype=np.float64)
    if patch.dither not in _DITHERS:
        raise ValueError(f"dither '{patch.dither}' desconocido. Disponibles: {sorted(_DITHERS)}")
    return _DITHERS[patch.dither](gray)


def render_wedge(
    patches: list[PatchSpec],
    total_w: int,
    total_h: int,
    *,
    label_each: bool = True,
    label_size_mm: float = 2.0,
    dpi: int = 169,
) -> np.ndarray:
    """
    Renderiza el step-wedge completo como binario uint8 (0=negro, 255=blanco).

    Fondo blanco; cada parche dithereado en su posicion; etiquetas opcionales
    sobre fondo blanco a la izquierda/abajo del parche para no contaminar el area
    medible.
    """
    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)
    for p in patches:
        block = render_patch(p)
        canvas[p.y_px:p.y_px + p.h_px, p.x_px:p.x_px + p.w_px] = block

    if label_each:
        # Etiquetas: las renderizamos con PIL.Draw en una banda blanca aparte y
        # las pegamos al canvas. Las etiquetas van DEBAJO de cada parche, fuera del
        # area del parche (para no afectar la medicion fotografica).
        label_px = _mm_to_px(label_size_mm, dpi)
        img = Image.fromarray(canvas, mode="L")
        draw = ImageDraw.Draw(img)
        font = _try_default_font(max(8, label_px))
        for p in patches:
            text = p.label
            # Posicion: justo debajo del parche, alineado izquierda
            x = p.x_px
            y = p.y_px + p.h_px + 2  # 2 px de gap
            if y + label_px > total_h:
                # No cabe debajo, no etiquetar
                continue
            try:
                draw.text((x, y), text, fill=0, font=font)
            except (OSError, ValueError):
                pass
        canvas = np.array(img, dtype=np.uint8)
    return canvas


def save_wedge_outputs(
    canvas: np.ndarray,
    patches: list[PatchSpec],
    out_png: Path,
    *,
    dpi: int,
    material: str | None,
    notes: str = "",
) -> dict:
    """Guarda PNG 1-bit + metadata JSON. Returns el dict de metadata."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(out_png, optimize=True)

    meta = {
        "schema": "laser_calibration_wedge/v1",
        "dpi": int(dpi),
        "material": material or "",
        "notes": notes,
        "interval_mm": 25.4 / float(dpi),
        "n_patches": len(patches),
        "patches": [asdict(p) for p in patches],
        "image_path": str(out_png),
        "image_size_px": [int(canvas.shape[1]), int(canvas.shape[0])],
        "image_size_mm": [
            round(canvas.shape[1] * 25.4 / dpi, 3),
            round(canvas.shape[0] * 25.4 / dpi, 3),
        ],
    }
    meta_path = out_png.parent / (out_png.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Genera step-wedge para calibracion fisica de material laser")
    p.add_argument("--out", type=Path, required=True, help="Ruta PNG de salida (1-bit ready para grabar)")
    p.add_argument("--steps", type=int, default=16, help="Numero de parches tonales (default 16)")
    p.add_argument("--square-mm", type=float, default=10.0, help="Lado de cada parche cuadrado (mm)")
    p.add_argument("--gap-mm", type=float, default=3.0, help="Separacion entre parches (mm)")
    p.add_argument("--margin-mm", type=float, default=5.0, help="Margen blanco alrededor (mm)")
    p.add_argument("--dpi", type=int, default=169, help="DPI del grabado (default 169 = max para spot 0.15 mm)")
    p.add_argument("--cols", type=int, default=0, help="Columnas de la grilla (0 = auto sqrt(steps))")
    p.add_argument(
        "--dither",
        choices=("floyd", "bayer8", "blue_noise_vac32", "threshold"),
        default="floyd",
        help="Algoritmo de dither por parche (constante en todo el wedge).",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma para distribuir valores de gris (>1 concentra oscuros; util para acrilico bajo potencia).",
    )
    p.add_argument("--no-labels", action="store_true", help="No imprimir etiquetas (parches limpios)")
    p.add_argument("--label-size-mm", type=float, default=2.5, help="Tamano de fuente etiqueta (mm)")
    p.add_argument(
        "--material",
        type=str,
        default="",
        help="Si se pasa nombre de MaterialProfile (laser_physics), valida DPI contra spot y guarda metadata.",
    )
    p.add_argument(
        "--material-presets-dir",
        type=Path,
        default=None,
        help="Directorio con JSONs custom de MaterialProfile.",
    )
    p.add_argument("--notes", type=str, default="", help="Notas libres anexadas al meta.")
    return p


def main() -> int:
    args = build_argument_parser().parse_args()

    cols = int(args.cols) if args.cols > 0 else None

    # Validacion contra material si se proporciona
    material_name: str | None = None
    if args.material:
        try:
            import laser_physics
            profile = laser_physics.load_material_profile(args.material, presets_dir=args.material_presets_dir)
            material_name = profile.name
            print(
                f"[MATERIAL] {profile.name} spot={profile.spot_mm:.3f}mm "
                f"default_dpi={profile.default_dpi} tone={profile.tone_response}",
                flush=True,
            )
            warn = profile.validate_dpi(args.dpi, emit_warning=False)
            if warn:
                print(f"[MATERIAL] WARN: {warn}", flush=True)
        except (ImportError, KeyError, ValueError, FileNotFoundError) as exc:
            print(f"[MATERIAL] omitido ({exc})", file=sys.stderr)

    patches, total_w, total_h = layout_patches(
        steps=int(args.steps),
        square_mm=float(args.square_mm),
        gap_mm=float(args.gap_mm),
        margin_mm=float(args.margin_mm),
        dpi=int(args.dpi),
        cols=cols,
        dither=str(args.dither),
        gamma=float(args.gamma),
    )

    print(
        f"[WEDGE] steps={len(patches)} grid={total_w}x{total_h}px "
        f"({total_w*25.4/args.dpi:.1f}x{total_h*25.4/args.dpi:.1f}mm) dither={args.dither}",
        flush=True,
    )

    canvas = render_wedge(
        patches, total_w, total_h,
        label_each=not args.no_labels,
        label_size_mm=float(args.label_size_mm),
        dpi=int(args.dpi),
    )

    meta = save_wedge_outputs(
        canvas, patches, args.out, dpi=int(args.dpi),
        material=material_name, notes=str(args.notes),
    )
    print(f"[WEDGE] PNG: {args.out}", flush=True)
    print(f"[WEDGE] meta: {args.out.parent / (args.out.stem + '_meta.json')}", flush=True)
    print(f"[WEDGE] interval mm = {meta['interval_mm']:.4f} (configurar en CAM)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
