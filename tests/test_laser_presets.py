"""Tests del sistema de presets + auto-detector (`scripts/laser_presets.py`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SCRIPTS = ROOT / "scripts"


def _load(name: str):
    if str(SCRIPT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPT_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPT_SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_catalog_has_all_expected_presets():
    lp = _load("laser_presets")
    names = {p.name for p in lp.ALL_PRESETS}
    assert "photo_general" in names
    assert "portrait" in names
    assert "scene_dark" in names
    assert "scene_bright" in names
    assert "poster_back_engrave" in names
    assert "line_art" in names


def test_get_preset_by_name():
    lp = _load("laser_presets")
    p = lp.get_preset("portrait")
    assert p.name == "portrait"
    assert p.algorithm == "stucki_serpentine"


def test_get_preset_unknown_raises():
    lp = _load("laser_presets")
    with pytest.raises(KeyError):
        lp.get_preset("nonexistent_preset")


def test_list_presets_dict_serializable():
    lp = _load("laser_presets")
    items = lp.list_presets_dict()
    assert isinstance(items, list) and len(items) >= 5
    for item in items:
        assert "name" in item and "label" in item and "params" in item
        params = item["params"]
        for k in ("algorithm", "threshold", "contrast", "brightness", "gamma", "autocontrast", "sharpen", "invert"):
            assert k in params


def test_compute_image_stats_uniform_gray():
    """Imagen uniforme: std=0, edge_density~0."""
    lp = _load("laser_presets")
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    s = lp.compute_image_stats(arr)
    assert s.mean == pytest.approx(128.0, abs=0.5)
    assert s.std < 1.0
    assert s.edge_density < 0.01


def test_compute_image_stats_high_contrast_bimodal():
    """Imagen mitad negro / mitad blanco: extreme_ratio alto, edge_density localizado."""
    lp = _load("laser_presets")
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :, :] = 0
    arr[64:, :, :] = 255
    s = lp.compute_image_stats(arr)
    assert s.extreme_ratio > 0.95
    assert s.std > 100


def test_recommend_poster_for_bimodal_high_contrast():
    """Imagen bimodal (mucho negro + mucho blanco) + bordes finos densos → poster preset."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(0)
    # Simular gráfico: bandas alternantes de negro/blanco (alto edge_density por las bandas)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 8):
        arr[i:i+4, :, :] = 255  # bandas blancas finas → muchos bordes
    # Inyectar algo de texto-like ruido para variar
    for k in range(rng.integers(15, 25)):
        y, x = rng.integers(0, 250, size=2)
        v = int(rng.choice([0, 255]))
        arr[y:y+4, x:x+30, :] = v
    r = lp.recommend_preset(arr)
    assert r.preset_name in ("poster_back_engrave", "line_art"), f"got {r.preset_name}"


def test_recommend_scene_dark_for_dark_image():
    """Foto oscura realista: mean bajo (~40), pero std alto por sombras/midtones."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(1)
    # Gradiente oscuro 5..90 con variación de textura
    h, w = 128, 128
    base = np.linspace(5, 90, h, dtype=np.float64)[:, None] + np.zeros((h, w), dtype=np.float64)
    noise = rng.normal(0, 22, size=(h, w))
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    r = lp.recommend_preset(arr)
    assert r.preset_name == "scene_dark", f"got {r.preset_name} stats={r.stats}"
    assert "oscura" in r.reason.lower()


def test_recommend_scene_bright_for_bright_image():
    """Foto clara realista: mean alto (~200+), std alto por highlights/midtones."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(2)
    h, w = 128, 128
    base = np.linspace(155, 250, h, dtype=np.float64)[:, None] + np.zeros((h, w), dtype=np.float64)
    noise = rng.normal(0, 28, size=(h, w))
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    r = lp.recommend_preset(arr)
    assert r.preset_name == "scene_bright", f"got {r.preset_name} stats={r.stats}"


def test_recommend_photo_general_for_balanced_image():
    """Foto natural balanceada (gradientes, variación tonal moderada) → photo_general."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(3)
    # Gradiente con ruido
    y_grad = np.linspace(80, 180, 128, dtype=np.float64)[:, None]
    x_grad = np.linspace(0, 30, 128, dtype=np.float64)[None, :]
    base = (y_grad + x_grad).clip(0, 255).astype(np.uint8)
    noise = rng.integers(-20, 21, size=(128, 128), dtype=np.int16)
    gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    r = lp.recommend_preset(arr)
    assert r.preset_name == "photo_general", f"got {r.preset_name}"


def test_recommendation_returns_image_stats():
    lp = _load("laser_presets")
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    r = lp.recommend_preset(arr)
    assert hasattr(r, "stats")
    assert hasattr(r.stats, "mean")
    assert hasattr(r.stats, "std")
    assert hasattr(r.stats, "extreme_ratio")
    assert hasattr(r.stats, "edge_density")


def test_preset_params_ranges_sane():
    """Sanity: cada preset tiene params dentro de rangos válidos."""
    lp = _load("laser_presets")
    for p in lp.ALL_PRESETS:
        assert 1 <= p.threshold <= 254, f"{p.name}: threshold {p.threshold}"
        assert 0.1 <= p.contrast <= 3.0, f"{p.name}: contrast {p.contrast}"
        assert -100 <= p.brightness <= 100, f"{p.name}: brightness {p.brightness}"
        assert 0.3 <= p.gamma <= 3.0, f"{p.name}: gamma {p.gamma}"
        assert 0 <= p.autocontrast <= 20, f"{p.name}: autocontrast {p.autocontrast}"
        assert 0 <= p.sharpen <= 300, f"{p.name}: sharpen {p.sharpen}"
