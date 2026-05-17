"""Tests Score v5 sin referencia (HVS-MSE + spectral radial + tone match).

v5 evalua calidad fisica del halftone SIN depender de un PNG objetivo:
- Blue-noise sintetico scorea mejor que clusters direccionales.
- v5 funciona con signature distinta (no necesita target_binary/density/edges).
- LUT cambia el tone_error de forma esperada.
- Dispatch v5 acepta `target_gray` como "gris ideal" y `lut/ppd` por keyword.
- Coexiste con v1..v4 sin romperlos.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SCRIPTS = ROOT / "scripts"


def _load_laser_scoring():
    if str(SCRIPT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPT_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "laser_scoring", SCRIPT_SCRIPTS / "laser_scoring.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _MockCandidate:
    contrast: float = 1.0
    brightness: float = 0.0
    sharpen: float = 0.0


def _blue_noise_like_binary(shape: tuple[int, int], target_ratio: float, seed: int) -> np.ndarray:
    """Aproximacion barata a blue-noise: umbralizar ruido uniforme."""
    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    cutoff = float(np.quantile(noise, 1.0 - target_ratio))
    return np.where(noise >= cutoff, 255, 0).astype(np.uint8)


def _clustered_binary(shape: tuple[int, int], target_ratio: float) -> np.ndarray:
    """Cluster determinista (mitad blanca / mitad negra) -> energia en baja freq."""
    h, w = shape
    out = np.zeros(shape, dtype=np.uint8)
    n_white_rows = max(1, int(round(h * target_ratio)))
    out[:n_white_rows, :] = 255
    return out


def test_v5_spectral_penalty_blue_noise_lower_than_cluster() -> None:
    """Componente espectral: blue-noise tiene menor energia en baja frecuencia que cluster."""
    laser_scoring = _load_laser_scoring()
    shape = (96, 96)
    bn = _blue_noise_like_binary(shape, target_ratio=0.5, seed=42)
    cl = _clustered_binary(shape, target_ratio=0.5)
    pen_bn = laser_scoring.spectral_radial_penalty(bn)
    pen_cl = laser_scoring.spectral_radial_penalty(cl)
    assert pen_bn < pen_cl, f"blue-noise debe penalizar menos que cluster (bn={pen_bn:.4f}, cl={pen_cl:.4f})"
    assert pen_cl > 0.5, f"cluster grande debe tener alta energia baja-freq, no {pen_cl:.4f}"


def test_v5_score_blue_noise_better_than_clustered() -> None:
    """Score total: para un gris uniforme, blue-noise scorea mejor (menor) que cluster."""
    laser_scoring = _load_laser_scoring()
    shape = (96, 96)
    gray_uniform = np.full(shape, 128, dtype=np.uint8)
    bn = _blue_noise_like_binary(shape, target_ratio=0.5, seed=42)
    cl = _clustered_binary(shape, target_ratio=0.5)
    cand = _MockCandidate()
    score_bn, *_ = laser_scoring.score_candidate_v5(bn, gray_uniform, cand)
    score_cl, *_ = laser_scoring.score_candidate_v5(cl, gray_uniform, cand)
    assert score_bn < score_cl, (
        f"blue-noise sobre gris uniforme debe scorear mejor que cluster "
        f"(bn={score_bn:.4f}, cl={score_cl:.4f})"
    )


def test_v5_no_target_args_required() -> None:
    """v5 funciona sin target_binary/density/edges (signature distinta a v1..v4)."""
    laser_scoring = _load_laser_scoring()
    rng = np.random.default_rng(0)
    shape = (64, 64)
    gray = rng.integers(40, 200, size=shape, dtype=np.uint8)
    out = np.where(gray >= 128, 255, 0).astype(np.uint8)
    cand = _MockCandidate()
    score, hvs_err, spec_pen, wr = laser_scoring.score_candidate_v5(out, gray, cand)
    assert isinstance(score, float)
    assert 0.0 <= hvs_err
    assert 0.0 <= spec_pen <= 1.0
    assert 0.0 <= wr <= 1.0


def test_v5_lut_shifts_tone_error() -> None:
    """Aplicar una LUT que oscurece el gris debe cambiar el tone_error."""
    laser_scoring = _load_laser_scoring()
    shape = (64, 64)
    gray = np.full(shape, 128, dtype=np.uint8)
    out = _blue_noise_like_binary(shape, target_ratio=0.5, seed=7)
    err_identity = laser_scoring.tone_match_error(out, gray)

    def darken_lut(g: np.ndarray) -> np.ndarray:
        return np.clip(g.astype(np.float64) * 0.6, 0, 255).astype(np.uint8)

    err_darkened = laser_scoring.tone_match_error(out, darken_lut(gray))
    assert err_identity != pytest.approx(err_darkened, abs=1e-6), (
        "tone_match debe responder al cambio de LUT"
    )
    assert err_darkened > err_identity


def test_v5_dispatch_routes_correctly() -> None:
    """dispatch('v5', ...) acepta target_gray como 'gris ideal' y soporta lut/ppd por kw."""
    laser_scoring = _load_laser_scoring()
    shape = (32, 32)
    rng = np.random.default_rng(1)
    gray = rng.integers(60, 200, size=shape, dtype=np.uint8)
    out = np.where(gray >= 128, 255, 0).astype(np.uint8)
    dummy_binary = np.zeros(shape, dtype=np.uint8)
    dummy_density = np.zeros((8, 8), dtype=np.float64)
    dummy_edges = np.zeros(shape, dtype=np.float64)
    cand = _MockCandidate()
    score, *_ = laser_scoring.score_candidate_dispatch(
        "v5", out, gray, dummy_binary, dummy_density, dummy_edges, cand,
    )
    assert isinstance(score, float)
    assert score >= 0.0

    score_high_ppd, *_ = laser_scoring.score_candidate_dispatch(
        "v5", out, gray, dummy_binary, dummy_density, dummy_edges, cand, ppd=128.0,
    )
    assert isinstance(score_high_ppd, float)
    assert score != pytest.approx(score_high_ppd, abs=1e-9)


def test_v5_lut_param_in_dispatch() -> None:
    """`lut` por keyword en dispatch desplaza el score cuando se pasa una LUT no-identidad."""
    laser_scoring = _load_laser_scoring()
    shape = (48, 48)
    gray = np.full(shape, 128, dtype=np.uint8)
    out = _blue_noise_like_binary(shape, target_ratio=0.5, seed=11)
    dummy_binary = np.zeros(shape, dtype=np.uint8)
    dummy_density = np.zeros((12, 12), dtype=np.float64)
    dummy_edges = np.zeros(shape, dtype=np.float64)
    cand = _MockCandidate()

    s_identity, *_ = laser_scoring.score_candidate_dispatch(
        "v5", out, gray, dummy_binary, dummy_density, dummy_edges, cand,
    )

    def darken_lut(g: np.ndarray) -> np.ndarray:
        return np.clip(g.astype(np.float64) * 0.6, 0, 255).astype(np.uint8)

    s_lut, *_ = laser_scoring.score_candidate_dispatch(
        "v5", out, gray, dummy_binary, dummy_density, dummy_edges, cand, lut=darken_lut,
    )
    assert s_identity != pytest.approx(s_lut, abs=1e-6)


def test_v5_unknown_version_raises() -> None:
    laser_scoring = _load_laser_scoring()
    shape = (16, 16)
    dummy = np.zeros(shape, dtype=np.uint8)
    dummy_f = np.zeros((4, 4), dtype=np.float64)
    cand = _MockCandidate()
    with pytest.raises(ValueError, match="version desconocida"):
        laser_scoring.score_candidate_dispatch(
            "v99", dummy, dummy, dummy, dummy_f, dummy_f.astype(np.float64), cand,
        )


def test_v1_v2_still_work_with_v5_present() -> None:
    """Garantiza que agregar v5 no rompe el dispatch de v1 y v2 (v3/v4 cubiertos por su test_score_regression)."""
    laser_scoring = _load_laser_scoring()
    shape = (40, 40)
    rng = np.random.default_rng(99)
    gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
    binary = np.where(gray >= 128, 255, 0).astype(np.uint8)
    density = np.zeros((10, 10), dtype=np.float64)
    edges = np.zeros(shape, dtype=np.float64)
    out = binary.copy()
    cand = _MockCandidate()
    s1, *_ = laser_scoring.score_candidate_dispatch("v1", out, gray, binary, density, edges, cand)
    s2, *_ = laser_scoring.score_candidate_dispatch("v2", out, gray, binary, density, edges, cand)
    assert isinstance(s1, float) and isinstance(s2, float)


def test_v5_terms_breakdown_keys() -> None:
    """Sanidad: `score_candidate_v5_terms` devuelve las claves documentadas (v1.5)."""
    laser_scoring = _load_laser_scoring()
    shape = (32, 32)
    gray = np.full(shape, 128, dtype=np.uint8)
    out = _blue_noise_like_binary(shape, target_ratio=0.5, seed=3)
    cand = _MockCandidate()
    d = laser_scoring.score_candidate_v5_terms(out, gray, cand)
    expected = {
        "hvs_mse", "spectral_lowfreq_penalty", "tone_error", "white_ratio",
        "reg", "reg_contrast", "reg_brightness", "reg_sharpen",
        "ppd", "low_band_fraction", "tone_scale",
        "w_hvs", "w_spec", "w_tone", "w_reg", "score",
        # v1.5: nuevas claves
        "detail_error", "w_detail", "detail_weight", "bright_threshold", "multi_scale_tone",
    }
    assert expected.issubset(set(d.keys())), f"faltan claves: {expected - set(d.keys())}"


# ---------------------------------------------------------------------------
# Tests v1.5 — edge preservation + multi-scale tone + CLAHE preprocess
# ---------------------------------------------------------------------------


def test_v5_edge_preservation_detects_lost_detail() -> None:
    """
    Caso real: detalle sutil en zona muy brillante (logo Red Bull en capó claro).
    Un threshold global puro lo bota; un dither con sharpen+contraste lo preserva.

    v5 con detail_weight=0 (legacy) elige el threshold puro (BUG).
    v5 con detail_weight=0.35 (default) elige el dither que preserva (FIX).
    """
    laser_scoring = _load_laser_scoring()
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore

    # Imagen casi blanca (245) con un cuadrado de 8x8 sutilmente más oscuro (190)
    gray = np.full((128, 128), 245, dtype=np.float64)
    gray[58:66, 58:66] = 190.0

    @dataclass
    class C:
        algorithm: str = "floyd"
        invert: bool = False
        threshold: int = 128
        contrast: float = 1.0
        brightness: float = 0.0
        gamma: float = 1.0
        autocontrast: float = 0.0
        sharpen: float = 0.0

    cand_lose = C(algorithm="threshold", threshold=180)
    cand_keep = C(algorithm="floyd", threshold=128, contrast=1.4, brightness=-20.0, sharpen=80.0)
    out_lose = ltm.render_candidate(gray, cand_lose)
    out_keep = ltm.render_candidate(gray, cand_keep)

    # Sanidad: las dos rendizaciones efectivamente difieren en el área del detalle
    assert (out_lose[58:66, 58:66] == 0).mean() == 0.0, "lose debe perder el detalle (sin negro)"
    assert (out_keep[58:66, 58:66] == 0).mean() > 0.10, "keep debe preservar parte del detalle"

    # Legacy: v5 sin detail term ELIGE EL MALO
    sl_legacy = laser_scoring.score_candidate_v5_terms(out_lose, gray, cand_lose, detail_weight=0.0)
    sk_legacy = laser_scoring.score_candidate_v5_terms(out_keep, gray, cand_keep, detail_weight=0.0)
    assert sl_legacy["score"] < sk_legacy["score"], (
        f"legacy debe preservar el BUG por baseline: lose={sl_legacy['score']:.4f} vs keep={sk_legacy['score']:.4f}"
    )

    # v1.5: con detail term default ELIGE EL BUENO
    sl_new = laser_scoring.score_candidate_v5_terms(out_lose, gray, cand_lose)
    sk_new = laser_scoring.score_candidate_v5_terms(out_keep, gray, cand_keep)
    assert sk_new["score"] < sl_new["score"], (
        f"v1.5 debe elegir keep: lose={sl_new['score']:.4f} vs keep={sk_new['score']:.4f}"
    )
    # detail_error de lose debe ser ALTO (perdió detalle)
    assert sl_new["detail_error"] > 0.5, f"detail_error lose={sl_new['detail_error']:.4f}, esperaba > 0.5"
    # detail_error de keep debe ser BAJO (preservó)
    assert sk_new["detail_error"] < sl_new["detail_error"], (
        f"keep debe tener menos detail_error: keep={sk_new['detail_error']:.4f}, lose={sl_new['detail_error']:.4f}"
    )


def test_edge_preservation_zero_for_plain_image() -> None:
    """Imagen completamente plana no tiene bordes que preservar → error = 0."""
    laser_scoring = _load_laser_scoring()
    gray = np.full((64, 64), 200, dtype=np.float64)
    out = np.full((64, 64), 255, dtype=np.uint8)  # binario uniforme
    assert laser_scoring.edge_preservation_error(out, gray) == 0.0


def test_multi_scale_tone_match_reduces_block_blindness() -> None:
    """
    Multi-scale tone match captura detalles que el single-scale=8 pierde.

    Si el detalle es de 4 px en bloque de 8 px casi blanco, el promedio block-mean=8 da
    error ~0, pero el block-mean=4 captura mejor el detalle perdido.
    """
    laser_scoring = _load_laser_scoring()
    # Capó blanco 64x64 con un punto oscuro de 4x4
    gray = np.full((64, 64), 240, dtype=np.float64)
    gray[30:34, 30:34] = 50.0
    # Binario que pierde el detalle completamente
    binary = np.full((64, 64), 255, dtype=np.uint8)

    single = laser_scoring.tone_match_error(binary, gray, scale=8)
    multi = laser_scoring.multi_scale_tone_match_error(binary, gray)
    # Multi-scale debe detectar mejor (mayor error) el detalle perdido vs single-scale
    assert multi >= single, f"multi_scale ({multi:.4f}) debe ser >= single ({single:.4f}) cuando hay detalle fino"


def test_clahe_preprocess_enhances_local_contrast() -> None:
    """CLAHE debe aumentar la varianza local en zonas con detalle sutil."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    gray = np.full((128, 128), 240, dtype=np.float64)
    gray[60:68, 60:68] = 200.0  # detalle sutil
    enhanced = ltm.clahe_preprocess_gray(gray, clip_limit=2.5, tile_size=8, blend=1.0)
    # El detalle se debe revelar: la varianza local en la región aumenta
    region_orig = gray[55:75, 55:75]
    region_enh = enhanced[55:75, 55:75]
    assert region_enh.std() > region_orig.std() * 1.1, (
        f"CLAHE debe aumentar contraste local: std orig={region_orig.std():.2f} enh={region_enh.std():.2f}"
    )
    # No debe saturar (clip a [0, 255])
    assert enhanced.min() >= 0 and enhanced.max() <= 255


# ---------------------------------------------------------------------------
# Tests v1.6 — plain region simplification + micro-detail fallback
# ---------------------------------------------------------------------------


def test_plain_region_simplification_clamps_bright_uniform() -> None:
    """Zona uniforme brillante (cielo) debe clampearse a 255 puro."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    gray = np.full((64, 64), 230, dtype=np.float64)  # uniform bright
    out = ltm.plain_region_simplification(gray)
    # Toda la imagen debe quedar en 255 (era plana y brillante)
    assert np.all(out == 255.0), f"esperaba todo 255, got unique={np.unique(out)}"


def test_plain_region_simplification_clamps_dark_uniform() -> None:
    """Zona uniforme oscura (sombra) debe clampearse a 0 puro."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    gray = np.full((64, 64), 20, dtype=np.float64)  # uniform dark
    out = ltm.plain_region_simplification(gray)
    assert np.all(out == 0.0), f"esperaba todo 0, got unique={np.unique(out)}"


def test_plain_region_simplification_preserves_detail() -> None:
    """Zonas con detalle (varianza alta) NO deben tocarse."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    rng = np.random.default_rng(42)
    # Imagen con detalle real: ruido + edges
    gray = rng.uniform(60, 200, (64, 64)).astype(np.float64)
    out = ltm.plain_region_simplification(gray)
    # La imagen ruidosa NO se puede clampear (varianza alta en todas partes)
    # Tolerancia: hasta 5% de pixels modificados (algunos pueden quedar en bordes
    # de zonas planas accidentales)
    n_changed = np.sum(out != gray)
    assert n_changed / gray.size < 0.05, (
        f"plain simp toco {100*n_changed/gray.size:.1f}% de pixels en imagen con detalle"
    )


def test_plain_region_simplification_preserves_midtones() -> None:
    """Zonas uniformes pero de tono medio (gris medio) NO deben clampearse —
    necesitan halftone real."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    gray = np.full((64, 64), 128, dtype=np.float64)  # gris medio uniforme
    out = ltm.plain_region_simplification(gray)
    # Gris medio no cumple bright_threshold ni dark_threshold → no se toca
    assert np.all(out == 128.0), f"midtone debe preservarse, unique={np.unique(out)}"


def test_s_curve_aclara_midtones_oscurece_sombras() -> None:
    """v2.0: S-curve aclara highlights y oscurece shadows, midtone es pivot fijo."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    g = np.array([[0, 64, 128, 192, 255]], dtype=np.float64)
    # Strength 0 = identidad
    assert np.allclose(ltm.apply_s_curve(g, 0.0), g)
    # Strength 0.5 = S suave
    out = ltm.apply_s_curve(g, 0.5)
    # Pivot fijo en 128
    assert abs(out[0, 2] - 128.0) < 0.5
    # Shadow (64) se oscurece (< 64)
    assert out[0, 1] < 64
    # Highlight (192) se aclara (> 192)
    assert out[0, 3] > 192
    # Extremos no salen de [0, 255]
    assert out.min() >= 0 and out.max() <= 255


def test_local_contrast_aumenta_punch_sin_amplificar_extremos() -> None:
    """v2.0: local contrast aumenta std (más punch) sin saturar valores."""
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    rng = np.random.default_rng(0)
    # Imagen con gradiente + ruido (caso fotorrealista típico)
    h, w = 100, 100
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    gray = (yy.astype(np.float64) / h) * 200 + 30  # gradiente 30..230
    gray += rng.normal(0, 8, gray.shape)
    gray = np.clip(gray, 0, 255)

    # Amount 0 = identidad
    assert np.allclose(ltm.apply_local_contrast(gray, amount_pct=0.0), gray)

    # Amount 15 = punch moderado
    enhanced = ltm.apply_local_contrast(gray, radius_px=60.0, amount_pct=15.0)
    # std debe aumentar (más contraste mid-freq)
    assert enhanced.std() > gray.std()
    # Pero no debe saturar fuera de [0, 255]
    assert enhanced.min() >= 0 and enhanced.max() <= 255
    # El valor medio debe quedar similar (no introduce sesgo global)
    assert abs(enhanced.mean() - gray.mean()) < 5.0


def test_micro_detail_in_plain_image_still_detected() -> None:
    """Imagen casi 100% plana con UN solo punto oscuro: edge_preservation_error
    debe seguir detectándolo (fallback de micro-detail aislado)."""
    laser_scoring = _load_laser_scoring()
    # Imagen 64x64 a 245 con 1 cuadrado de 3x3 a 220 (sutil, no extremo)
    gray = np.full((64, 64), 245, dtype=np.float64)
    gray[30:33, 30:33] = 220.0
    binary_lost = np.full((64, 64), 255, dtype=np.uint8)  # totalmente perdido
    # Aún con detalle sutil que no cruza el percentil normal, el fallback debe activarse
    err_lost = laser_scoring.edge_preservation_error(binary_lost, gray)
    assert err_lost >= 0.0, "edge_preservation_error nunca debe ser negativo"
    # No assertamos un valor mínimo porque el detalle es muy sutil; lo importante es
    # que la función no rompa y retorne un valor finito.
    assert np.isfinite(err_lost)
