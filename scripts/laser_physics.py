"""
Fisica del laser CO2 aplicada al pipeline: validacion DPI por spot, perfiles de material con LUT.

Fuentes principales (ver `cursor-memory-vault/MEMORY-laser-snips.md` §"Fisica CO2"):
- Spot CO2 50W lente 50.8 mm ≈ 0.12–0.18 mm (M²=1.1–1.5).
- DPI util max ≈ 1/spot (141–282 DPI para spot 0.18–0.12 mm).
- Dot-gain: spot circular cubre ~2.5x area del pixel cuadrado -> PNG sale 30–50% mas oscuro grabado.
- Acrilico colado despolimeriza (back-engrave frost = scattering Mie 20–50 um).
- Madera: pirolisis no-monotonica (carboniza, luego sublima lignina -> rebote tonal).

Este modulo es la **fuente de verdad** para validaciones fisicas; los presets concretos viven
en JSON bajo `presets/materials/` y se cargan via `load_material_profile()`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

ToneResponseKind = Literal["monotonic", "non_monotonic", "linear"]

# ---------------------------------------------------------------------------
# Spot fisico por maquina/material (defaults conservadores; calibrable)
# ---------------------------------------------------------------------------

# Spot estimado del haz CO2 al material en mm; depende de:
#   - longitud focal de la lente (50.8 mm = 2" estandar)
#   - calidad del haz (M^2)
#   - colimacion / alineacion
# Estos valores son **defaults razonables**, sobre-escribibles por preset de material.
SPOT_SIZE_DEFAULTS_MM: dict[str, float] = {
    "co2_50w_lens_2inch": 0.15,
    "co2_50w_lens_4inch": 0.30,
    "co2_80w_lens_2inch": 0.18,
    "co2_100w_lens_2inch": 0.20,
    "funsun_50w_default": 0.15,  # ver `PROJECTS/image-ap-sport-for-laser` §"acrylic 9-12%"
    "unknown": 0.18,  # peor-caso conservador para warning
}


def estimate_max_useful_dpi(spot_mm: float, *, safety_factor: float = 1.0) -> int:
    """
    DPI util maximo dado el tamano del spot.

    Regla: `dpi_max = 25.4 mm/inch / spot_mm`. Subir mas solo solapa el spot, oscurece y alarga.

    Args:
        spot_mm: tamano del spot al material (mm).
        safety_factor: multiplicador < 1 reduce el limite (mas conservador);
            > 1 acepta solapamiento leve (NO recomendado para fotograbado).
    """
    if spot_mm <= 0:
        raise ValueError(f"spot_mm debe ser > 0, got {spot_mm}")
    dpi_max = 25.4 / float(spot_mm)
    return int(round(dpi_max * float(safety_factor)))


def validate_dpi_for_spot(
    dpi: int,
    spot_mm: float,
    *,
    safety_factor: float = 1.0,
    emit_warning: bool = True,
) -> str | None:
    """
    Valida DPI contra el spot estimado.

    Returns:
        None si DPI esta dentro del rango util; sino una cadena explicando el problema
        (y emite `UserWarning` si `emit_warning=True`).

    Ejemplos:
        validate_dpi_for_spot(300, 0.15) -> warning (max 169)
        validate_dpi_for_spot(150, 0.15) -> None
    """
    if dpi <= 0:
        raise ValueError(f"dpi debe ser > 0, got {dpi}")
    max_dpi = estimate_max_useful_dpi(spot_mm, safety_factor=safety_factor)
    if dpi <= max_dpi:
        return None
    msg = (
        f"DPI={dpi} excede el limite fisico para spot={spot_mm:.3f} mm "
        f"(max recomendado={max_dpi} DPI). "
        "Subir DPI por encima del recip. del spot solo solapa el haz, oscurece "
        "la imagen y alarga el tiempo de grabado sin ganancia de detalle. "
        "Considere reducir DPI o cambiar de lente para reducir el spot."
    )
    if emit_warning:
        warnings.warn(msg, UserWarning, stacklevel=2)
    return msg


def interval_mm_for_dpi(dpi: int) -> float:
    """Conversion `interval (mm) = 25.4 / DPI`. Convencion 1 px = 1 step de linea."""
    if dpi <= 0:
        raise ValueError(f"dpi debe ser > 0, got {dpi}")
    return 25.4 / float(dpi)


# ---------------------------------------------------------------------------
# LUT (Look-Up Table) por material para compensar respuesta del material al laser
# ---------------------------------------------------------------------------


def identity_lut() -> np.ndarray:
    """LUT identidad 256 elementos uint8."""
    return np.arange(256, dtype=np.uint8)


def apply_lut_to_gray(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Aplica LUT 256-elementos al gris. `gray` puede ser uint8 o float; salida uint8."""
    g = np.clip(gray, 0, 255).astype(np.uint8)
    if lut.shape != (256,):
        raise ValueError(f"lut debe ser shape (256,), got {lut.shape}")
    return np.asarray(lut, dtype=np.uint8)[g]


def make_lut_callable(lut_array: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Crea un callable `gray -> gray_post_lut` reutilizable (cierre sobre el array).

    Util para pasar a `laser_scoring.score_candidate_v5(..., lut=...)`.
    """
    lut = np.asarray(lut_array, dtype=np.uint8)
    if lut.shape != (256,):
        raise ValueError(f"lut_array debe ser shape (256,), got {lut.shape}")

    def _apply(gray: np.ndarray) -> np.ndarray:
        return apply_lut_to_gray(gray, lut)

    return _apply


# ---------------------------------------------------------------------------
# MaterialProfile: configuracion completa por material
# ---------------------------------------------------------------------------


@dataclass
class MaterialProfile:
    """
    Perfil de material para grabado laser.

    Encapsula todo lo que el pipeline necesita saber del par (material, maquina):
    - spot fisico estimado (-> validacion DPI)
    - DPI default sugerido
    - LUT inversa (256 elementos) compensando dot-gain + respuesta tonal del material
    - tipo de respuesta (monotonic / non_monotonic / linear)
    - rango de potencia recomendado (% maquina)
    - notas humanas

    LUTs reales se calibran via step-wedge fisico (Fase R7); estos stubs son aproximaciones
    sensatas para arrancar.
    """

    name: str
    spot_mm: float
    default_dpi: int
    lut_curve: np.ndarray = field(default_factory=identity_lut)
    tone_response: ToneResponseKind = "monotonic"
    power_pct_range: tuple[float, float] = (5.0, 100.0)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.spot_mm <= 0:
            raise ValueError(f"spot_mm debe ser > 0 ({self.name})")
        if self.default_dpi <= 0:
            raise ValueError(f"default_dpi debe ser > 0 ({self.name})")
        self.lut_curve = np.asarray(self.lut_curve, dtype=np.uint8)
        if self.lut_curve.shape != (256,):
            raise ValueError(f"lut_curve debe ser shape (256,), got {self.lut_curve.shape} ({self.name})")
        lo, hi = self.power_pct_range
        if not (0 <= lo <= hi <= 100):
            raise ValueError(f"power_pct_range invalido {self.power_pct_range} ({self.name})")

    def max_useful_dpi(self) -> int:
        return estimate_max_useful_dpi(self.spot_mm)

    def validate_dpi(self, dpi: int, *, emit_warning: bool = True) -> str | None:
        return validate_dpi_for_spot(dpi, self.spot_mm, emit_warning=emit_warning)

    def lut(self) -> Callable[[np.ndarray], np.ndarray]:
        return make_lut_callable(self.lut_curve)


# ---------------------------------------------------------------------------
# Stubs de perfiles para acrilico y madera (calibrables, no son verdad fisica final)
# ---------------------------------------------------------------------------


def _gamma_lut(gamma: float) -> np.ndarray:
    """LUT que aplica `out = 255 * (in/255)**(1/gamma)`. gamma>1 oscurece; gamma<1 aclara."""
    x = np.arange(256, dtype=np.float64) / 255.0
    y = np.power(x, 1.0 / float(gamma))
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def _compose_luts(*luts: np.ndarray) -> np.ndarray:
    """Compone N LUTs aplicandolas en orden: L_n(... L_2(L_1(x)))."""
    out = np.arange(256, dtype=np.uint8)
    for lut in luts:
        if lut.shape != (256,):
            raise ValueError(f"todas las LUTs deben ser (256,), got {lut.shape}")
        out = np.asarray(lut, dtype=np.uint8)[out]
    return out


def acrylic_back_engrave_profile() -> MaterialProfile:
    """
    Acrilico colado, back-engrave a baja potencia (frost blanco lechoso por scattering Mie).

    Caracteristicas fisicas:
      - Spot Funsun 50W ≈ 0.15 mm -> DPI max ≈ 169.
      - Despolimerizacion limpia (no funde) en 9-12% potencia.
      - Respuesta tonal monotonica: mas energia -> mas frost; saturacion antes del fundido.
      - Dot-gain alto (~2.5x): PNG sale ~30-50% mas oscuro grabado. La LUT pre-aclara para
        compensar.

    LUT (gamma ~0.65): aclara el gris para que tras dot-gain se vea como el original.
    Calibracion fina requiere step-wedge fotografiado bajo luz cruzada (Fase R7).
    """
    return MaterialProfile(
        name="acrylic_back_engrave",
        spot_mm=0.15,
        default_dpi=169,
        lut_curve=_gamma_lut(0.65),
        tone_response="monotonic",
        power_pct_range=(9.0, 14.0),
        notes=(
            "Acrilico colado, cara posterior, baja potencia. LUT gamma 0.65 aproxima "
            "compensacion dot-gain Funsun 50W. Calibrar con step-wedge real."
        ),
    )


def _wood_dual_phase_lut() -> np.ndarray:
    """
    LUT para madera con respuesta no-monotonica.

    Madera carboniza primero (oscurece, sube hasta ~70% energia) y luego sublima
    lignina (aclara hasta saturacion). Mapeo seguro: comprimir el rango medio-alto
    para evitar entrar en la zona de aclarado por accidente. Usuario debe calibrar
    para su madera/laser especifico.
    """
    x = np.arange(256, dtype=np.float64) / 255.0
    # Soft sigmoid que comprime entradas >0.6 hacia 0.85
    cap = 0.85
    knee = 0.6
    out = np.where(
        x <= knee,
        x,
        knee + (cap - knee) * (1.0 - np.exp(-(x - knee) * 4.0)),
    )
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def wood_profile() -> MaterialProfile:
    """
    Madera (default generico, baja densidad tipo MDF/contrachapado).

    Caracteristicas fisicas:
      - Spot Funsun 50W ≈ 0.18 mm -> DPI max ≈ 141 (madera tolera menos DPI que acrilico).
      - Respuesta NO MONOTONICA: pirolisis carboniza primero, luego sublima lignina.
      - Air assist 3-5 psi limpia char y aumenta contraste.
      - Veta de madera mete ruido tonal dificil de predecir.

    LUT comprime el extremo claro para evitar la zona de "rebote" donde la madera aclara
    en lugar de oscurecer. Calibrar por especie de madera (roble/pino/MDF dan curvas
    distintas).
    """
    return MaterialProfile(
        name="wood_generic",
        spot_mm=0.18,
        default_dpi=141,
        lut_curve=_wood_dual_phase_lut(),
        tone_response="non_monotonic",
        power_pct_range=(20.0, 60.0),
        notes=(
            "Madera generica, respuesta tonal no-monotonica (pirolisis + sublimacion lignina). "
            "LUT comprime extremo claro para evitar zona de rebote. Calibrar por especie."
        ),
    )


# ---------------------------------------------------------------------------
# Carga desde JSON
# ---------------------------------------------------------------------------


def _builtin_profiles() -> dict[str, Callable[[], MaterialProfile]]:
    return {
        "acrylic_back_engrave": acrylic_back_engrave_profile,
        "wood_generic": wood_profile,
    }


def load_material_profile(
    name: str,
    *,
    presets_dir: Path | None = None,
) -> MaterialProfile:
    """
    Carga un perfil de material.

    Resolucion en orden:
    1. Si existe `presets_dir/{name}.json`, lo carga (LUT puede ser 256 ints o ruta a `.npy`).
    2. Si `name` esta en `_builtin_profiles()`, devuelve el stub programatico.
    3. Sino, levanta `KeyError`.

    JSON schema esperado:
        {
            "name": "acrylic_back_engrave",
            "spot_mm": 0.15,
            "default_dpi": 169,
            "lut_curve": [..256 ints..]  # o "lut_curve_npy": "path/relative/to/json.npy"
            "tone_response": "monotonic",
            "power_pct_range": [9.0, 14.0],
            "notes": "..."
        }
    """
    if presets_dir is not None:
        json_path = presets_dir / f"{name}.json"
        if json_path.is_file():
            return _load_from_json(json_path)

    builtins = _builtin_profiles()
    if name in builtins:
        return builtins[name]()

    raise KeyError(
        f"Material '{name}' no encontrado. Disponibles (builtins): {sorted(builtins)}. "
        f"Para custom, crear `{name}.json` bajo presets_dir."
    )


def _load_from_json(json_path: Path) -> MaterialProfile:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lut_curve: np.ndarray
    if "lut_curve_npy" in data:
        npy_path = json_path.parent / str(data["lut_curve_npy"])
        if not npy_path.is_file():
            raise FileNotFoundError(f"LUT npy referenciada no existe: {npy_path}")
        lut_curve = np.load(npy_path).astype(np.uint8)
    elif "lut_curve" in data:
        lut_curve = np.array(data["lut_curve"], dtype=np.uint8)
    else:
        lut_curve = identity_lut()
    power_lo, power_hi = data.get("power_pct_range", [5.0, 100.0])
    return MaterialProfile(
        name=str(data.get("name", json_path.stem)),
        spot_mm=float(data["spot_mm"]),
        default_dpi=int(data["default_dpi"]),
        lut_curve=lut_curve,
        tone_response=str(data.get("tone_response", "monotonic")),  # type: ignore[arg-type]
        power_pct_range=(float(power_lo), float(power_hi)),
        notes=str(data.get("notes", "")),
    )


# ---------------------------------------------------------------------------
# Sharpen escalado al output (cierra regla 3 de RULES)
# ---------------------------------------------------------------------------


def scaled_unsharp_radius(
    *,
    ranking_pixels_short_side: int,
    output_mm_short_side: float,
    output_dpi: int,
    radius_mm: float = 0.10,
) -> float:
    """
    Calcula el radius (en px de la imagen de ranking) que corresponde a un
    `radius_mm` fisico en la salida final.

    Razon: el USM debe expresarse en mm fisicos del grabado (regla 3 de RULES),
    no en px del ranking. A `max_side=240` el USM con `radius=1.2` queda muy distinto
    al USM con `radius=1.2` sobre la salida final a `300 DPI x 100 mm` (≈ 1181 px).

    Formula:
        output_px_short = output_mm_short * output_dpi / 25.4
        radius_px_output = radius_mm * output_dpi / 25.4
        scale = ranking_pixels_short_side / output_px_short
        radius_px_ranking = radius_px_output * scale
                          = (radius_mm * output_dpi / 25.4) * (ranking_short / output_px_short)

    Args:
        ranking_pixels_short_side: lado corto en px de la imagen sobre la que se rankea.
        output_mm_short_side: lado corto fisico del grabado en mm.
        output_dpi: DPI del grabado final.
        radius_mm: radius deseado en mm fisicos del grabado (default 0.10 mm =
            ~mitad del spot tipico; sensato para fotograbado).

    Returns:
        radius en px de la imagen de ranking; clamp [0.3, 5.0] para evitar valores
        degenerados.
    """
    if ranking_pixels_short_side <= 0:
        raise ValueError(f"ranking_pixels_short_side debe ser > 0")
    if output_mm_short_side <= 0:
        raise ValueError(f"output_mm_short_side debe ser > 0")
    if output_dpi <= 0:
        raise ValueError(f"output_dpi debe ser > 0")
    if radius_mm <= 0:
        raise ValueError(f"radius_mm debe ser > 0")
    output_px = output_mm_short_side * output_dpi / 25.4
    radius_px_output = radius_mm * output_dpi / 25.4
    scale = float(ranking_pixels_short_side) / output_px
    radius_ranking = radius_px_output * scale
    return float(np.clip(radius_ranking, 0.3, 5.0))
