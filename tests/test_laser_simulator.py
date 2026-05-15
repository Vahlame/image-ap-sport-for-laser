"""Tests del simulador de grabado físico (`scripts/laser_simulator.py`)."""

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


def test_compute_spot_sigma_px_typical() -> None:
    sim = _load("laser_simulator")
    # spot 0.15 mm @ 169 DPI: interval = 25.4/169 ≈ 0.150 mm/px → spot_px = 1.0 → sigma ≈ 0.425
    sigma = sim.compute_spot_sigma_px(0.15, 169)
    assert 0.3 <= sigma <= 0.6, f"sigma inesperado {sigma}"


def test_compute_spot_sigma_px_invalid() -> None:
    sim = _load("laser_simulator")
    with pytest.raises(ValueError):
        sim.compute_spot_sigma_px(0.0, 169)
    with pytest.raises(ValueError):
        sim.compute_spot_sigma_px(0.15, 0)


def test_simulate_engraving_shape_and_dtype() -> None:
    sim = _load("laser_simulator")
    rng = np.random.default_rng(0)
    binary = (rng.random((128, 128)) > 0.5).astype(np.uint8) * 255
    out = sim.simulate_engraving(binary, spot_mm=0.15, output_dpi=169)
    assert out.shape == binary.shape
    assert out.dtype == np.uint8
    assert 0 <= out.min() and out.max() <= 255


def test_simulate_acrylic_frost_appearance() -> None:
    """Acrílico frost: zona grabada (binary=255) debe quedar más clara que fondo."""
    sim = _load("laser_simulator")
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[20:44, 20:44] = 255  # bloque grabado en el centro
    out = sim.simulate_engraving(binary, 0.15, 169, material_appearance="acrylic_frost")
    center = out[30:34, 30:34].mean()
    corner = out[0:4, 0:4].mean()
    assert center > corner + 50, f"frost no aclara el centro: center={center}, corner={corner}"


def test_simulate_wood_burn_appearance() -> None:
    """Madera burn: zona grabada debe quedar más oscura que el fondo."""
    sim = _load("laser_simulator")
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[20:44, 20:44] = 255
    out = sim.simulate_engraving(binary, 0.15, 169, material_appearance="wood_burn")
    center = out[30:34, 30:34].mean()
    corner = out[0:4, 0:4].mean()
    assert center < corner - 50, f"burn no oscurece el centro: center={center}, corner={corner}"


def test_simulate_raw_appearance_no_inversion() -> None:
    """Modo raw: la salida es el blur normalizado sin remapping."""
    sim = _load("laser_simulator")
    binary = np.zeros((32, 32), dtype=np.uint8)
    binary[10:22, 10:22] = 255
    out = sim.simulate_engraving(binary, 0.15, 169, material_appearance="raw")
    # raw: zona grabada queda clara (cerca de 255), borde algo gris, fondo cerca de 0
    center = out[14:18, 14:18].mean()
    corner = out[0:2, 0:2].mean()
    assert center > 200, f"center raw debe ser cerca de 255, got {center}"
    assert corner < 10, f"corner raw debe ser cerca de 0, got {corner}"


def test_simulate_invalid_appearance_raises() -> None:
    sim = _load("laser_simulator")
    binary = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="material_appearance"):
        sim.simulate_engraving(binary, 0.15, 169, material_appearance="nonsense")


def test_simulate_invalid_input_shape_raises() -> None:
    sim = _load("laser_simulator")
    with pytest.raises(ValueError, match="2D"):
        sim.simulate_engraving(np.zeros((8, 8, 3), dtype=np.uint8), 0.15, 169)


def test_simulate_from_material_profile_acrylic() -> None:
    sim = _load("laser_simulator")
    lp = _load("laser_physics")
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[20:44, 20:44] = 255
    profile = lp.acrylic_back_engrave_profile()
    out = sim.simulate_from_material_profile(binary, profile)
    # Acrílico → frost → centro mas claro que esquinas
    assert out[32, 32] > out[0, 0] + 50


def test_simulate_from_material_profile_wood() -> None:
    sim = _load("laser_simulator")
    lp = _load("laser_physics")
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[20:44, 20:44] = 255
    profile = lp.wood_profile()
    out = sim.simulate_from_material_profile(binary, profile)
    # Madera → wood_burn → centro más oscuro que esquinas
    assert out[32, 32] < out[0, 0] - 50


def test_simulate_higher_dpi_less_blur() -> None:
    """A mayor DPI con mismo spot, el blur relativo (en pixeles) es mayor.

    Spot 0.15 mm físico:
    - 169 DPI → interval ≈ 0.150 mm → spot ≈ 1.0 px → sigma ≈ 0.43 px (poco blur)
    - 600 DPI → interval ≈ 0.042 mm → spot ≈ 3.5 px → sigma ≈ 1.51 px (más blur)

    Más DPI sin reducir spot = más solapamiento físico = más blur perceptual.
    """
    sim = _load("laser_simulator")
    sigma_low = sim.compute_spot_sigma_px(0.15, 169)
    sigma_high = sim.compute_spot_sigma_px(0.15, 600)
    assert sigma_high > sigma_low * 2


def test_simulate_isolated_pixel_dot_gain() -> None:
    """
    Un pixel aislado encendido debe convertirse en un dot circular con dot-gain
    visible (NO queda como un solo pixel blanco neto).
    """
    sim = _load("laser_simulator")
    binary = np.zeros((32, 32), dtype=np.uint8)
    binary[16, 16] = 255  # 1 pixel encendido
    out = sim.simulate_engraving(binary, 0.30, 169, material_appearance="raw")
    # vecinos del pixel central deben tener valor > 0 (dot-gain del blur)
    neighbor_mean = out[15:18, 15:18].mean()
    assert neighbor_mean > 5, f"dot-gain no se aprecia: neighbor_mean={neighbor_mean}"
