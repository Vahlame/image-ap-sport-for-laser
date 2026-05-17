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
from PIL import Image


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
    # v2.0+: técnicas profesionales (PhotoGrav/Photoshop workflow) — default 0=sin efecto
    s_curve_strength: float = 0.0       # 0=identidad, 0.4-0.5 suave, 1.0 agresiva
    local_contrast_amount: float = 0.0   # 0=sin, 10-15 típico para fotorrealismo

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
            "s_curve_strength": self.s_curve_strength,
            "local_contrast_amount": self.local_contrast_amount,
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
    s_curve_strength=0.5,        # v2.0: aclara midtones, oscurece sombras (workflow Photoshop/PhotoGrav)
    local_contrast_amount=12.0,  # v2.0: clarity moderada para "punch" fotorrealista
)

# Preset CONSERVADOR para fotos con detalles finos (texto, pelaje, cabello, bokeh).
# El default agresivo `photo_back_engrave` perdía texto chico y texturas finas porque
# gamma 1.55 + sharpen 130 colapsa los grises medios. Este preset:
#   - gamma 1.20: cerca de lineal, preserva midtones donde está el detalle fino
#   - sharpen 80: realza bordes pero sin amplificar ruido
#   - threshold 110: deja más densidad de puntos negros (más fidelidad al original)
#   - autocontrast 1.5: contraste leve, no satura
#   - CLAHE preprocess: redistribuye contraste local para revelar detalle en zonas extremas
#   - jarvis_serpentine: error diffusion balanceado, mejor para detalle fino que stucki+sharpen
PRESET_PHOTO_HIGH_DETAIL = LaserPreset(
    name="photo_high_detail",
    label="Foto con detalle fino (texto, pelaje, cabello)",
    description="Foto con elementos detallados que el preset default `photo_back_engrave` "
                "podría saturar/perder: texto pequeño, pelaje, cabello, bokeh con luces puntuales. "
                "Usa CLAHE + jarvis_serpentine + sharpen moderado + gamma cercano a lineal "
                "para preservar fidelidad en midtones donde vive el detalle. Recomendado "
                "automáticamente cuando edge_density > 0.12.",
    algorithm="jarvis_serpentine",
    preprocess_mode="clahe",
    threshold=110,
    contrast=1.10,
    brightness=0.0,
    gamma=1.20,
    autocontrast=1.5,
    sharpen=80.0,
    invert=True,
    suggested_material="acrylic_funsun_9060_back_engrave",
    s_curve_strength=0.4,        # v2.0: S-curve suave (preset conservador, no aplastar más)
    local_contrast_amount=10.0,  # v2.0: clarity leve para detalle fino sin amplificar ruido
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


# Preset para anime/illustration/cartoon sobre acrílico back-engrave.
# Caso típico: render de personaje anime (Miku, etc.), logo con sujeto + fondo blanco,
# dibujo digital con outlines y colores planos.
#
# Workflow del usuario:
#   - Fondo blanco puro → NO grabar (transparente al ver desde el frente)
#   - Sujeto entero (cara + pelo + ropa con todos sus colores) → grabar sólido como FROST
#
# Diferencia clave vs photo_back_engrave (halftone):
#   - Halftone convierte midtones en puntos → el sujeto se ve "puntillista"
#   - Threshold puro convierte midtones en blanco/negro sólido → el sujeto se ve SÓLIDO
#
# Implementación:
#   - algorithm="threshold" (sin halftone, binario puro)
#   - threshold=240 (alto: solo pixels CASI BLANCOS son "fondo")
#   - invert=True (fondo blanco → 0 = no grabar, resto → 255 = frost)
#   - preprocess "none" (no necesitamos suavizar/contrastar — el dibujo ya tiene contraste)
#   - sharpen 0 (no agregar bordes — el anime ya los tiene definidos)
# Preset OPTIMIZADO para texturas finas (textiles, mallas, tejidos houndstooth,
# follaje fino, plumas, pelaje denso). El preset photo_high_detail prioriza
# nitidez perceptual; este preset prioriza FIDELIDAD A TEXTURAS REPETITIVAS.
#
# Diferencias vs photo_high_detail:
#   - Algorithm: stucki_serpentine (vs jarvis) — Stucki distribuye error en
#     mayor área (12 vecinos vs 7) → patrones repetitivos se preservan mejor
#   - sharpen=120 (vs 80) — realza más los bordes finos
#   - threshold=120 (vs 110) — un poco más blanco para que las texturas se
#     vean como puntos finos sobre fondo claro
#   - autocontrast=2.0 (vs 1.5) — más contraste para que las texturas resalten
#   - preprocess_mode="sauvola" (vs CLAHE) — sauvola contrastea localmente
#     SIN amplificar ruido de bokehs/gradientes
#   - DOC: usuario debe desactivar plain_region_simplification para evitar que
#     el algoritmo confunda texturas finas con "zonas planas"
PRESET_PHOTO_FINE_TEXTURES = LaserPreset(
    name="photo_fine_textures",
    label="Foto con texturas finas (textiles, follaje, pelaje denso)",
    description="Optimizado para fotos donde el VALOR está en preservar texturas finas "
                "repetitivas: tejidos houndstooth, mallas, follaje, plumas, pelaje denso. "
                "Stucki serpentine + sharpen alto + sauvola local. Recomendado cuando "
                "edge_density > 10% AND std > 75 AND extreme_ratio < 50% (típico de fotos "
                "con elemento textil/orgánico dominante). IMPORTANTE: desactivar "
                "simplify_plain_regions porque las texturas pueden malinterpretarse como "
                "regiones uniformes.",
    algorithm="stucki_serpentine",
    preprocess_mode="sauvola",
    threshold=120,
    contrast=1.20,
    brightness=0.0,
    gamma=1.10,
    autocontrast=2.0,
    sharpen=120.0,
    invert=True,
    suggested_material="acrylic_funsun_9060_back_engrave",
    s_curve_strength=0.3,        # v2.0: S-curve leve (las texturas ya tienen su propio contraste)
    local_contrast_amount=15.0,  # v2.0: clarity media para realzar el patrón repetitivo
)


PRESET_CARTOON_BACK_ENGRAVE = LaserPreset(
    name="cartoon_back_engrave",
    label="Anime/illustration sobre acrílico (silueta sólida)",
    description="Dibujo anime/manga/cartoon con fondo blanco grande. Convierte el sujeto entero "
                "en silueta sólida frost (sin halftone) — el fondo blanco queda transparente. "
                "Recomendado cuando very_bright_ratio > 50% y extreme_ratio > 65% (mucho "
                "fondo blanco puro). "
                "IMPORTANTE: como `preprocess_gray` aplica `invert` ANTES del threshold, "
                "para detectar 'fondo blanco puro' usamos threshold=10 + invert=True: "
                "el fondo (gray=255) se invierte a 0 → queda bajo el threshold → NO grabar; "
                "cualquier sujeto no-blanco (gray<245) se invierte a >10 → cruza threshold → "
                "GRABAR como frost sólido.",
    algorithm="threshold",
    preprocess_mode="none",
    threshold=10,
    contrast=1.0,
    brightness=0.0,
    gamma=1.0,
    autocontrast=0.0,
    sharpen=0.0,
    invert=True,
    suggested_material="acrylic_funsun_9060_back_engrave",
)


ALL_PRESETS: tuple[LaserPreset, ...] = (
    PRESET_PHOTO_GENERAL,
    PRESET_PORTRAIT,
    PRESET_SCENE_DARK,
    PRESET_SCENE_BRIGHT,
    PRESET_POSTER_BACK_ENGRAVE,
    PRESET_PHOTO_BACK_ENGRAVE,
    PRESET_PHOTO_HIGH_DETAIL,
    PRESET_PHOTO_FINE_TEXTURES,
    PRESET_LINE_ART,
    PRESET_CARTOON_BACK_ENGRAVE,
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
    # v1.7+: very_bright_ratio detecta caso "sujeto midtone + fondo brillante grande"
    # (mujer kayak: 21% cielo+lago) que confunde a extreme_ratio.
    very_bright_ratio: float = 0.0  # fracción pixeles > 215 (subset de extreme_ratio)


_DETECTOR_NORMALIZE_SIZE = 800  # lado mayor en px para normalizar antes de calcular stats


def _normalize_for_stats(rgb: np.ndarray) -> np.ndarray:
    """
    Resize la imagen a max_side=800 antes de calcular stats. Esto hace que edge_density
    sea size-invariant: Sobel sobre una imagen 8000x6000 da un edge_density menor que
    sobre una imagen 700x520 de la misma escena, porque el threshold del gradient
    (>0.15) captura solo bordes "fuertes" que son más visibles tras downsampling.

    Sin esta normalización, el detector elegía presets distintos según el tamaño del
    upload (caso real: Hokusai 2000px daba edge=5%, 700px daba edge=10.8%).
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= _DETECTOR_NORMALIZE_SIZE:
        return rgb
    scale = _DETECTOR_NORMALIZE_SIZE / long_side
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    img = Image.fromarray(rgb)
    return np.array(img.resize((new_w, new_h), Image.Resampling.LANCZOS))


def compute_image_stats(rgb: np.ndarray) -> ImageStats:
    """Calcula estadísticos rápidos sobre la imagen RGB (size-normalized a max_side=800)."""
    rgb = _normalize_for_stats(rgb)
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
    return ImageStats(
        mean=mean,
        std=std,
        extreme_ratio=extreme,
        edge_density=edge_density,
        very_bright_ratio=very_light,
    )


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

    # Si el material es acrílico back-engrave, decisión basada en composición:
    #
    # Regla 0 (v1.8) — Anime/illustration con fondo blanco grande → cartoon_back_engrave
    #   Detección: very_bright_ratio > 50% AND extreme_ratio > 65% — fondo predominantemente
    #   blanco puro (>215) + alta bimodalidad → es un dibujo con sujeto sobre fondo limpio.
    #   El sujeto entero debe grabar como silueta sólida frost (sin halftone puntillista).
    #
    # Regla A — Foto rica en midtones (extreme_ratio < 40%) → photo_high_detail
    #   Validado: cervatillo bokeh (extr=8%), libro texto (extr=24%), cachorrito (extr=37%).
    #
    # Regla B (v1.7) — Caso "sujeto midtone + fondo muy brillante" → photo_high_detail
    #   Detección: very_bright_ratio > 18% (mucho cielo/lago/fondo blanco) Y
    #   extreme_ratio < 75% (no es bimodal puro tipo poster).
    #   Caso real: mujer kayak (vbright=21%, extr=66%) — el preset agresivo aplastaba
    #   cabello/ropa midtone a blanco aunque el extreme_ratio alto sugería bimodal.
    #
    # Regla C — Foto pure-bimodal (Earth extr=45% vbright=7%, rally car extr=47% vbright≈10%,
    #   Jupiter extr=41% vbright=4%, hands extr=53%) → photo_back_engrave default.
    if is_acrylic:
        if s.very_bright_ratio > 0.50 and s.extreme_ratio > 0.65:
            return Recommendation(
                preset_name=PRESET_CARTOON_BACK_ENGRAVE.name,
                preset_label=PRESET_CARTOON_BACK_ENGRAVE.label,
                reason=(
                    f"Dibujo/anime/illustration con fondo blanco grande "
                    f"({s.very_bright_ratio*100:.0f}% blanco puro, extremos {s.extreme_ratio*100:.0f}%). "
                    f"Threshold puro sin halftone convierte el sujeto entero en silueta sólida frost "
                    f"(el fondo blanco queda transparente al grabar en acrílico back-engrave)."
                ),
                stats=s,
            )
        # Regla A.5 (v1.9): texturas finas dominantes — edge_density alto + std moderado-alto
        # + NO bimodal (extr < 50%). Típico de fotos con textil/follaje/pelaje denso o
        # ilustraciones tradicionales con muchas líneas finas (Hokusai, etc.).
        if s.edge_density > 0.10 and s.std > 50 and s.extreme_ratio < 0.50:
            return Recommendation(
                preset_name=PRESET_PHOTO_FINE_TEXTURES.name,
                preset_label=PRESET_PHOTO_FINE_TEXTURES.label,
                reason=(
                    f"Foto con texturas finas dominantes "
                    f"(bordes {s.edge_density*100:.0f}%, std {s.std:.0f}) — Stucki + sharpen alto "
                    f"para preservar tejidos/follaje/pelaje. Tip: en Modo Manual desactivá "
                    f"'Simplificar fondos planos' si las texturas se ven planas."
                ),
                stats=s,
            )
        if s.extreme_ratio < 0.40:
            return Recommendation(
                preset_name=PRESET_PHOTO_HIGH_DETAIL.name,
                preset_label=PRESET_PHOTO_HIGH_DETAIL.label,
                reason=(
                    f"Foto rica en midtones (extremos solo {s.extreme_ratio*100:.0f}%) "
                    f"sobre acrílico — preset conservador con CLAHE + gamma 1.2 + sharpen 80 "
                    f"para preservar texto/pelaje/cabello/bokeh que el preset default "
                    f"saturaría aplastándolos a blanco."
                ),
                stats=s,
            )
        if s.very_bright_ratio > 0.18 and s.extreme_ratio < 0.75:
            return Recommendation(
                preset_name=PRESET_PHOTO_HIGH_DETAIL.name,
                preset_label=PRESET_PHOTO_HIGH_DETAIL.label,
                reason=(
                    f"Foto con fondo muy brillante ({s.very_bright_ratio*100:.0f}% blancos puros, "
                    f"como cielo/lago/pared) y sujeto midtone — preset conservador para preservar "
                    f"el detalle del sujeto que el preset agresivo aplastaría a blanco."
                ),
                stats=s,
            )
        return Recommendation(
            preset_name=PRESET_PHOTO_BACK_ENGRAVE.name,
            preset_label=PRESET_PHOTO_BACK_ENGRAVE.label,
            reason=(
                f"Foto bimodal sobre acrílico back-engrave (extremos {s.extreme_ratio*100:.0f}%, "
                f"material='{material}'). Usa preset con contraste/sharpen/gamma altos y "
                f"threshold bajo para que los dots del halftone queden definidos al grabar a DPI bajo."
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
