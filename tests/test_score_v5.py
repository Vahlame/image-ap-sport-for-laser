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
    """Sanidad: `score_candidate_v5_terms` devuelve las claves documentadas."""
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
    }
    assert expected.issubset(set(d.keys())), f"faltan claves: {expected - set(d.keys())}"
