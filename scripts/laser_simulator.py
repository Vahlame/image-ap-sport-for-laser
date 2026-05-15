#!/usr/bin/env python3
"""
Simulador de grabado físico: predice cómo se verá el PNG 1-bit tras grabar.

Modelo simple pero útil:
1. **Dot-gain del spot**: el haz láser tiene un diámetro físico (`spot_mm`); cada
   pixel "encendido" del PNG produce una marca circular gaussiana de ese diámetro.
   En píxeles: `sigma_px ≈ spot_px / 2.355` (full-width-half-max).
2. **Respuesta tonal del material**: tras el blur del spot, se mapea la intensidad
   acumulada a la apariencia visible:
   - **Acrílico back-engrave**: zonas grabadas se ven como FROST BLANCO sobre
     fondo (transparente) negro. Sin LUT inverso adicional, la salida representa
     directamente la luminancia que vería el observador.
   - **Madera / cuero**: zonas grabadas se ven como CARBONIZADO OSCURO sobre
     fondo (madera) claro. Convención inversa: la imagen aparece como un
     positivo oscuro sobre claro.
3. **Convención del PNG de entrada**: `255` = láser ENCENDIDO (graba), `0` = OFF.

Salida: PNG en mode `L` (uint8 0..255) que representa **cómo se vería el grabado**
fotografiado bajo luz difusa. NO es el archivo a enviar al CAM (ese sigue siendo
el PNG 1-bit binario).

Uso CLI:
    python scripts/laser_simulator.py \\
        --input runs/_agricultor_2026-05-15/agricultor_hi_contrast_v4_0.3247.png \\
        --out simulated.png --material acrylic_back_engrave --output-dpi 169

Uso programático:
    from laser_simulator import simulate_engraving
    sim = simulate_engraving(binary_png, profile, output_dpi=169)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import laser_physics  # noqa: E402


# Constante: relación FWHM ↔ sigma para una gaussiana
# FWHM = 2 * sqrt(2 ln 2) * sigma ≈ 2.355 * sigma
FWHM_TO_SIGMA = 1.0 / 2.355


def compute_spot_sigma_px(spot_mm: float, output_dpi: int) -> float:
    """
    Convierte el spot físico (mm) a sigma gaussiano en píxeles, dado el DPI de salida.

    El spot se interpreta como FWHM del haz; sigma = FWHM / 2.355.

    Args:
        spot_mm: diámetro del spot en mm (e.g., 0.15 para Funsun 50W).
        output_dpi: DPI del grabado final (i.e., 25.4 / interval_mm).

    Returns:
        sigma del filtro Gaussian en píxeles del PNG.
    """
    if spot_mm <= 0:
        raise ValueError(f"spot_mm debe ser > 0, got {spot_mm}")
    if output_dpi <= 0:
        raise ValueError(f"output_dpi debe ser > 0, got {output_dpi}")
    interval_mm = 25.4 / float(output_dpi)
    spot_px = float(spot_mm) / interval_mm
    sigma = spot_px * FWHM_TO_SIGMA
    return max(0.3, sigma)  # piso 0.3 px para que siempre haya algo de blur


def simulate_engraving(
    binary_png: np.ndarray,
    spot_mm: float,
    output_dpi: int,
    *,
    material_appearance: str = "acrylic_frost",
    background_value: int | None = None,
) -> np.ndarray:
    """
    Simula la apariencia del grabado físico a partir del PNG 1-bit.

    Args:
        binary_png: array (H, W) uint8 con valores {0, 255}. 255 = láser ON.
        spot_mm: diámetro físico del spot.
        output_dpi: DPI del grabado.
        material_appearance: cómo presentar el grabado:
            - "acrylic_frost": blanco sobre fondo oscuro (back-engrave acrílico).
            - "wood_burn": oscuro sobre fondo claro (madera / cuero).
            - "raw": float blurred sin normalizar (debug).
        background_value: opcional override del fondo (0..255). Si None, usa default por modo.

    Returns:
        array (H, W) uint8 representando la luminancia simulada vista por el ojo.
    """
    if binary_png.ndim != 2:
        raise ValueError(f"binary_png debe ser 2D, got shape {binary_png.shape}")
    if material_appearance not in ("acrylic_frost", "wood_burn", "raw"):
        raise ValueError(f"material_appearance desconocido: {material_appearance}")

    # Normalizar a [0, 1]: 1 donde el láser graba
    img01 = (binary_png.astype(np.float64) / 255.0)
    sigma = compute_spot_sigma_px(spot_mm, output_dpi)
    # Aplicar blur gaussiano (simula la marca circular del spot por cada pixel encendido)
    blurred = gaussian_filter(img01, sigma=sigma, mode="nearest")
    # Renormalizar: el blur conserva energía pero como cada "punto" se distribuye,
    # el peak puede caer. Para que zonas saturadas (mucho blanco) lleguen a ~1:
    # no aplicamos renormalización, dejamos el blur tal cual — esto reproduce
    # fielmente el dot-gain (zonas dispersas se ven más grises).
    blurred = np.clip(blurred, 0.0, 1.0)

    if material_appearance == "raw":
        return np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

    if material_appearance == "acrylic_frost":
        # Frost blanco sobre fondo oscuro (vista normal del acrílico back-engrave)
        bg = 18 if background_value is None else int(background_value)
        fg = 245  # max frost achievable (no es 255 puro porque no hay 100% diffuse)
        out = bg + blurred * (fg - bg)
    else:  # "wood_burn"
        # Carbonizado oscuro sobre fondo claro (madera natural)
        bg = 200 if background_value is None else int(background_value)
        fg = 35  # max char depth
        out = bg - blurred * (bg - fg)

    return np.clip(out, 0, 255).astype(np.uint8)


def simulate_from_material_profile(
    binary_png: np.ndarray,
    profile: "laser_physics.MaterialProfile",
    output_dpi: int | None = None,
) -> np.ndarray:
    """
    Conveniencia: usa la apariencia derivada del MaterialProfile.

    Si `output_dpi` es None, usa `profile.default_dpi`.
    """
    appearance = _appearance_from_profile(profile)
    dpi = int(output_dpi) if output_dpi else int(profile.default_dpi)
    return simulate_engraving(
        binary_png, spot_mm=profile.spot_mm, output_dpi=dpi,
        material_appearance=appearance,
    )


def _appearance_from_profile(profile: "laser_physics.MaterialProfile") -> str:
    """Mapea heurísticamente el MaterialProfile a un material_appearance."""
    name = profile.name.lower()
    if "acrylic" in name or "back_engrave" in name:
        return "acrylic_frost"
    return "wood_burn"  # default conservador para madera/cuero/oscuros


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simula la apariencia del grabado físico desde un PNG 1-bit laser-ready."
    )
    p.add_argument("--input", type=Path, required=True, help="PNG 1-bit input (0/255).")
    p.add_argument("--out", type=Path, required=True, help="Ruta del PNG simulado.")
    p.add_argument(
        "--material",
        type=str,
        default="",
        help="Nombre MaterialProfile (e.g. 'acrylic_back_engrave'). Si vacío, usar --spot-mm y --appearance.",
    )
    p.add_argument(
        "--material-presets-dir",
        type=Path,
        default=None,
        help="Directorio con JSONs custom de MaterialProfile.",
    )
    p.add_argument("--spot-mm", type=float, default=0.15, help="Spot físico (sin --material).")
    p.add_argument(
        "--output-dpi",
        type=int,
        default=169,
        help="DPI del grabado (default 169 = 1/spot para 0.15 mm).",
    )
    p.add_argument(
        "--appearance",
        choices=("acrylic_frost", "wood_burn", "raw"),
        default="acrylic_frost",
        help="Apariencia material (ignorado si --material se pasa).",
    )
    p.add_argument(
        "--background",
        type=int,
        default=-1,
        help="Override del fondo (0..255). -1 = default por modo.",
    )
    return p


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.input.is_file():
        print(f"[SIM] input no encontrado: {args.input}", file=sys.stderr)
        return 2

    binary = np.array(Image.open(args.input).convert("L"), dtype=np.uint8)
    if not set(np.unique(binary).tolist()).issubset({0, 255}):
        print(f"[SIM] WARN: input no es estrictamente 0/255 (valores: {len(np.unique(binary))} únicos); umbralizando >=128", file=sys.stderr)
        binary = np.where(binary >= 128, 255, 0).astype(np.uint8)

    if args.material:
        profile = laser_physics.load_material_profile(args.material, presets_dir=args.material_presets_dir)
        print(f"[SIM] material {profile.name} spot={profile.spot_mm:.3f}mm tone={profile.tone_response}", flush=True)
        sim = simulate_from_material_profile(binary, profile, output_dpi=int(args.output_dpi))
    else:
        bg = args.background if args.background >= 0 else None
        sim = simulate_engraving(
            binary, spot_mm=float(args.spot_mm), output_dpi=int(args.output_dpi),
            material_appearance=str(args.appearance), background_value=bg,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sim, mode="L").save(args.out, optimize=True)
    sigma = compute_spot_sigma_px(
        spot_mm=float(args.spot_mm) if not args.material else laser_physics.load_material_profile(args.material).spot_mm,
        output_dpi=int(args.output_dpi),
    )
    print(f"[SIM] sigma usado: {sigma:.3f} px (spot={(args.spot_mm if not args.material else 'profile'):s} mm @ {args.output_dpi} DPI)", flush=True)
    print(f"[SIM] guardado: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
