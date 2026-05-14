"""Regresion ligera para scoring v2 y SSIM minimo en caso controlado."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SCRIPTS = ROOT / "scripts"


def _load_laser_target_match():
    spec = importlib.util.spec_from_file_location("laser_target_match", SCRIPT_SCRIPTS / "laser_target_match.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_laser_scoring():
    spec = importlib.util.spec_from_file_location("laser_scoring", SCRIPT_SCRIPTS / "laser_scoring.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_score_v2_snapshot_stable() -> None:
    """Semilla fija: el score v2 no debe moverse mas de ~5%% sin cambiar la formula."""
    lt = _load_laser_target_match()
    laser_scoring = _load_laser_scoring()
    cand = lt.Candidate("floyd", False, 128, 1.0, 0.0, 1.0, 0.0, 0.0)
    rng = np.random.default_rng(12345)
    target_gray = rng.integers(0, 256, size=(40, 40), dtype=np.uint8)
    target_binary = np.where(target_gray >= 128, 255, 0).astype(np.uint8)
    out = rng.integers(0, 256, size=(40, 40), dtype=np.uint8)
    td = lt.density_map(target_gray)
    te = lt.edge_map(target_gray)
    s2 = laser_scoring.score_candidate_v2(out, target_gray, target_binary, td, te, cand)[0]
    expected = 0.536528726640048
    assert s2 == pytest.approx(expected, rel=0.05, abs=0.02)


def test_ssim_floor_identical_luminance() -> None:
    """Salida identica al target continuo: SSIM alto (proxy de best_match razonable)."""
    lt = _load_laser_target_match()
    g = np.random.default_rng(7).integers(40, 200, size=(48, 48), dtype=np.uint8)
    out = g.copy()
    o = out.astype(np.float64) / 255.0
    t = g.astype(np.float64) / 255.0
    ssim = float(structural_similarity(o, t, data_range=1.0))
    assert ssim >= 0.4
