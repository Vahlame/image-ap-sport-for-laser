#!/usr/bin/env python3
"""
Presets curados + auto-detector para que cualquier imagen salga con acabado impecable.

Diseño:
- Un preset = conjunto coherente de parámetros (algoritmo + tonales + preprocess) que
  produce buen resultado para una **clase** de imagen.
- `recommend_preset(rgb)` analiza estadísticos básicos de la imagen y devuelve el preset
  recomendado + razón.
- La UI usa "auto" por defecto: corre el detector y aplica el preset; el usuario puede
  cambiar a uno manual o tunear los sliders.

Heurísticos del detector (no ML — sólo histograma + bordes):

| Stat | Cómo se mide | Implica |
|---|---|---|
| `mean` | luminancia media (0..255) | <60 oscura, >180 clara |
| `std` | varianza tonal | <30 plana, >70 alto contraste |
| `extreme_ratio` | fracción pixeles <40 ó >215 | >0.5 bimodal (poster/gráfico) |
| `edge_density` | fracción gradientes >0.15 | >0.12 texto/líneas finas |

Reglas de decisión (en orden, primer match gana):

1. `extreme_ratio > 0.5` + `edge_density > 0.08` → **`poster_back_engrave`**
2. `edge_density > 0.18` + `std > 50` → **`line_art`**
3. `mean < 60` + `std > 30` → **`scene_dark`**
4. `mean > 180` + `std > 40` → **`scene_bright`**
5. fallback → **`photo_general`**
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LaserPreset:
    """Preset = conjunto coherente de params + identidad humana."""

    name: str
    label: str
    description: str
    # Params concretos del pipeline (subset de ProcessParams)
    algorithm: str
    preprocess_mode: str
    threshold: int
    contrast: float
    brightness: float
    gamma: float
    autocontrast: float
    sharpen: float
    invert: bool
    # Hint del material recomendado (UI lo sugiere; el usuario puede sobreescribir)
    suggested_material: str = ""

    def as_param_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "preprocess_mode": self.preprocess_mode,
            "threshold": self.threshold,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "gamma": self.gamma,
            "autocontrast": self.autocontrast,
            "sharpen": self.sharpen,
            "invert": self.invert,
        }


# ---------------------------------------------------------------------------
# Catálogo curado (validado experimentalmente sobre stock diverso)
# ---------------------------------------------------------------------------


PRESET_PHOTO_GENERAL = LaserPreset(
    name="photo_general",
    label="Foto general",
    description="Foto natural a color (paisajes, retratos de cuerpo, escenas). "
                "Jarvis serpentine + sauvola + invertido para polaridad CORRECTA: "
                "el sujeto oscuro de la foto se graba dark en madera (positivo).",
    algorithm="jarvis_serpentine",
    preprocess_mode="sauvola",
    threshold=128,
    contrast=1.15,
    brightness=0.0,
    gamma=1.0,
    autocontrast=1.5,
    sharpen=70.0,
    invert=True,
    suggested_material="wood_generic",
)

PRESET_PORTRAIT = LaserPreset(
    name="portrait",
    label="Retrato (persona/animal)",
    description="Primer plano de cara o animal. Stucki preserva texturas suaves de piel/pelaje. "
                "Invertido para grabado POSITIVO: cara/pelaje oscuro → wood burned dark, "
                "fondo claro → wood natural.",
    algorithm="stucki_serpentine",
    preprocess_mode="sauvola",
    threshold=128,
    contrast=1.10,
    brightness=5.0,
    gamma=1.05,
    autocontrast=1.0,
    sharpen=90.0,
    invert=True,
    suggested_material="wood_generic",
)

PRESET_SCENE_DARK = LaserPreset(
    name="scene_dark",
    label="Escena oscura",
    description="Foto oscura (noche, sombras, interior). Mantiene polaridad: highlights "
                "(luna/lámparas/destellos) se graban dark; áreas oscuras quedan naturales. "
                "Efecto 'constelación' artístico.",
    algorithm="jarvis_serpentine",
    preprocess_mode="sauvola",
    threshold=128,
    contrast=1.10,
    brightness=15.0,
    gamma=1.35,
    autocontrast=2.0,
    sharpen=65.0,
    invert=False,
    suggested_material="wood_generic",
)

PRESET_SCENE_BRIGHT = LaserPreset(
    name="scene_bright",
    label="Escena clara",
    description="Foto muy clara (cielo abierto, nieve, fondo blanco). Invertido para "
                "grabar SOLO los sujetos oscuros sobre fondo natural (positivo limpio).",
    algorithm="jarvis_serpentine",
    preprocess_mode="sauvola",
    threshold=118,
    contrast=1.20,
    brightness=-10.0,
    gamma=0.85,
    autocontrast=2.0,
    sharpen=75.0,
    invert=True,
    suggested_material="wood_generic",
)

PRESET_POSTER_BACK_ENGRAVE = LaserPreset(
    name="poster_back_engrave",
    label="Poster acrílico back-engrave",
    description="Diseño gráfico con texto/logos contrastados, destinado a grabar la cara posterior de "
                "acrílico (frost blanco sobre fondo transparente). Invertido + alto contraste.",
    algorithm="floyd",
    preprocess_mode="sauvola",
    threshold=75,
    contrast=1.0,
    brightness=10.0,
    gamma=1.2,
    autocontrast=2.0,
    sharpen=60.0,
    invert=True,
    suggested_material="acrylic_back_engrave",
)

# Preset NUEVO (v1.2): FOTO sobre acrilico back-engrave.
# El problema observado con poster_back_engrave aplicado a fotos: midtones excesivos
# -> frost muy parejo, sin definicion. Esta variante usa:
#   - stucki_serpentine: kernel grande con error mejor distribuido -> dots mas crisp
#   - gamma 1.55: aplasta midtones para que solo highlights/shadows fuertes generen frost
#   - threshold 105: balance frost/transparente para foto (no 75 que era para text)
#   - autocontrast 3.5: fuerza contraste sin perder informacion en extremos
#   - sharpen 120: realza bordes para que la foto no se vea borrosa al hacer halftone
#   - NO invert: para acrilico back-engrave, las luces de la foto = frost visible
#     (el usuario debe NO invertir tambien en LightBurn).
PRESET_PHOTO_BACK_ENGRAVE = LaserPreset(
    name="photo_back_engrave",
    label="Foto sobre acrílico back-engrave",
    description="Foto natural destinada a grabarse en la cara posterior de acrílico (sujeto visible "
                "como frost blanco al ver desde el frente, fondo transparente). Invertido para que "
                "el sujeto oscuro de la foto = white en PNG = frost. Stucki serpentine + gamma alta "
                "+ contraste fuerte + sharpen muy alto + threshold bajo para que los dots queden "
                "DEFINIDOS y no se vea borroso al grabar. NO subir invert/threshold en LightBurn — "
                "el PNG ya está listo para Pass-Through.",
    algorithm="stucki_serpentine",
    preprocess_mode="sauvola",
    threshold=95,
    contrast=1.35,
    brightness=0.0,
    gamma=1.55,
    autocontrast=3.5,
    sharpen=130.0,
    invert=True,
    suggested_material="acrylic_funsun_9060_back_engrave",
)

PRESET_LINE_ART = LaserPreset(
    name="line_art",
    label="Line art / vector",
    description="Dibujo a línea, logo vector, texto puro (líneas oscuras sobre fondo claro). "
                "Threshold sin dither + sharpen alto + invertido para grabar líneas dark.",
    algorithm="threshold",
    preprocess_mode="none",
    threshold=140,
    contrast=1.30,
    brightness=0.0,
    gamma=1.0,
    autocontrast=0.0,
    sharpen=110.0,
    invert=True,
    suggested_material="wood_generic",
)


ALL_PRESETS: tuple[LaserPreset, ...] = (
    PRESET_PHOTO_GENERAL,
    PRESET_PORTRAIT,
    PRESET_SCENE_DARK,
    PRESET_SCENE_BRIGHT,
    PRESET_POSTER_BACK_ENGRAVE,
    PRESET_PHOTO_BACK_ENGRAVE,
    PRESET_LINE_ART,
)

PRESETS_BY_NAME: dict[str, LaserPreset] = {p.name: p for p in ALL_PRESETS}


# ---------------------------------------------------------------------------
# Auto-detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageStats:
    mean: float
    std: float
    extreme_ratio: float  # fracción pixeles <40 ó >215
    edge_density: float  # fracción gradientes (Sobel) > 0.15


def compute_image_stats(rgb: np.ndarray) -> ImageStats:
    """Calcula estadísticos rápidos sobre la imagen RGB (resize previo a 256 lado corto recomendado)."""
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        r = rgb[..., 0].astype(np.float64)
        g = rgb[..., 1].astype(np.float64)
        b = rgb[..., 2].astype(np.float64)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif rgb.ndim == 2:
        gray = rgb.astype(np.float64)
    else:
        raise ValueError(f"shape inesperado: {rgb.shape}")

    mean = float(gray.mean())
    std = float(gray.std())
    very_dark = float((gray < 40).mean())
    very_light = float((gray > 215).mean())
    extreme = very_dark + very_light
    # gradient mag (Sobel-ish via np.gradient)
    gy, gx = np.gradient(gray / 255.0)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float((mag > 0.15).mean())
    return ImageStats(mean=mean, std=std, extreme_ratio=extreme, edge_density=edge_density)


@dataclass(frozen=True)
class Recommendation:
    preset_name: str
    preset_label: str
    reason: str
    stats: ImageStats


def recommend_preset(rgb: np.ndarray, *, material: str = "") -> Recommendation:
    """
    Decide el preset óptimo según estadísticos. Orden importa (primer match gana).

    Si `material` empieza con `"acrylic"`, las recomendaciones de fotos cambian:
    usa `photo_back_engrave` en vez de `photo_general`/`portrait`/`scene_*` — porque
    en acrílico el halftone necesita mucho más contraste/sharpen + spot mayor que
    en madera para que los dots se vean definidos al grabar.
    """
    s = compute_image_stats(rgb)
    is_acrylic = material.startswith("acrylic")

    # Regla 1: bimodal alto contraste + bordes finos → poster/gráfico
    if s.extreme_ratio > 0.5 and s.edge_density > 0.08:
        return Recommendation(
            preset_name=PRESET_POSTER_BACK_ENGRAVE.name,
            preset_label=PRESET_POSTER_BACK_ENGRAVE.label,
            reason=(
                f"Imagen bimodal de alto contraste "
                f"(extremos {s.extreme_ratio*100:.0f}%, bordes {s.edge_density*100:.0f}%) — "
                f"característico de gráficos/posters."
            ),
            stats=s,
        )

    # Regla 2: muy alta densidad de bordes + std alto → line art
    if s.edge_density > 0.18 and s.std > 50:
        return Recommendation(
            preset_name=PRESET_LINE_ART.name,
            preset_label=PRESET_LINE_ART.label,
            reason=(
                f"Mucha densidad de bordes ({s.edge_density*100:.0f}%) — "
                f"característico de line art / texto."
            ),
            stats=s,
        )

    # Si el material es acrílico back-engrave, TODAS las fotos van al preset acrílico-específico
    # (no usar scene_dark/bright que están tuneados para madera).
    if is_acrylic:
        return Recommendation(
            preset_name=PRESET_PHOTO_BACK_ENGRAVE.name,
            preset_label=PRESET_PHOTO_BACK_ENGRAVE.label,
            reason=(
                f"Foto sobre acrílico back-engrave (material='{material}'). "
                f"Usa preset con contraste/sharpen/gamma altos y threshold bajo "
                f"para que los dots del halftone queden definidos al grabar a DPI bajo."
            ),
            stats=s,
        )

    # Regla 3: escena oscura
    if s.mean < 60 and s.std > 30:
        return Recommendation(
            preset_name=PRESET_SCENE_DARK.name,
            preset_label=PRESET_SCENE_DARK.label,
            reason=f"Luminancia media baja ({s.mean:.0f}/255) — escena oscura.",
            stats=s,
        )

    # Regla 4: escena clara
    if s.mean > 180 and s.std > 30:
        return Recommendation(
            preset_name=PRESET_SCENE_BRIGHT.name,
            preset_label=PRESET_SCENE_BRIGHT.label,
            reason=f"Luminancia media alta ({s.mean:.0f}/255) — escena clara.",
            stats=s,
        )

    # Default: foto general
    return Recommendation(
        preset_name=PRESET_PHOTO_GENERAL.name,
        preset_label=PRESET_PHOTO_GENERAL.label,
        reason=(
            f"Foto natural (mean {s.mean:.0f}, std {s.std:.0f}, "
            f"bordes {s.edge_density*100:.0f}%)."
        ),
        stats=s,
    )


def get_preset(name: str) -> LaserPreset:
    """Devuelve preset por nombre. Lanza KeyError si no existe."""
    if name not in PRESETS_BY_NAME:
        raise KeyError(f"preset '{name}' desconocido. Disponibles: {sorted(PRESETS_BY_NAME)}")
    return PRESETS_BY_NAME[name]


def list_presets_dict() -> list[dict[str, Any]]:
    """Para serializar al cliente: lista de presets con sus campos."""
    return [
        {
            "name": p.name,
            "label": p.label,
            "description": p.description,
            "params": p.as_param_dict(),
            "suggested_material": p.suggested_material,
        }
        for p in ALL_PRESETS
    ]
