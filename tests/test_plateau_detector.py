"""Tests para PlateauDetector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_plateau():
    spec = importlib.util.spec_from_file_location("laser_plateau", ROOT / "scripts" / "laser_plateau.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plateau_detector_triggers_on_flat_window() -> None:
    lp = _load_plateau()
    det = lp.PlateauDetector(window_size=4, std_max=0.01)
    assert det.observe(1.0) == lp.PlateauAction.NONE
    assert det.observe(1.0) == lp.PlateauAction.NONE
    assert det.observe(1.0) == lp.PlateauAction.NONE
    assert det.observe(1.0) == lp.PlateauAction.RESTART
    assert det.observe(0.5) == lp.PlateauAction.NONE


def test_plateau_detector_reset_clears_window() -> None:
    lp = _load_plateau()
    det = lp.PlateauDetector(window_size=3, std_max=0.001)
    det.observe(2.0)
    det.observe(2.0)
    det.reset()
    assert det.observe(0.0) == lp.PlateauAction.NONE
