"""Tests para `scripts/laser_physics.py`: validacion DPI por spot, LUT material, sharpen escalado."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SCRIPTS = ROOT / "scripts"


def _load_laser_physics():
    if str(SCRIPT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPT_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "laser_physics", SCRIPT_SCRIPTS / "laser_physics.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_estimate_max_useful_dpi_funsun_50w() -> None:
    """Spot 0.15 mm -> DPI util max ≈ 169."""
    lp = _load_laser_physics()
    assert lp.estimate_max_useful_dpi(0.15) == 169
    assert lp.estimate_max_useful_dpi(0.18) == 141  # spot mas grande -> DPI menor
    assert lp.estimate_max_useful_dpi(0.12) == 212  # spot mas chico -> DPI mayor


def test_estimate_max_useful_dpi_invalid_spot() -> None:
    lp = _load_laser_physics()
    with pytest.raises(ValueError):
        lp.estimate_max_useful_dpi(0.0)
    with pytest.raises(ValueError):
        lp.estimate_max_useful_dpi(-1.0)


def test_validate_dpi_for_spot_ok() -> None:
    """DPI dentro del rango no devuelve mensaje."""
    lp = _load_laser_physics()
    assert lp.validate_dpi_for_spot(150, 0.15, emit_warning=False) is None
    assert lp.validate_dpi_for_spot(169, 0.15, emit_warning=False) is None


def test_validate_dpi_for_spot_warns_and_explains() -> None:
    """DPI > 1/spot devuelve string + emite UserWarning."""
    lp = _load_laser_physics()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        msg = lp.validate_dpi_for_spot(300, 0.15, emit_warning=True)
    assert msg is not None
    assert "300" in msg
    assert "0.150" in msg
    assert "169" in msg  # max DPI esperado
    assert any(issubclass(w.category, UserWarning) for w in captured)


def test_interval_mm_for_dpi() -> None:
    lp = _load_laser_physics()
    assert lp.interval_mm_for_dpi(300) == pytest.approx(0.08466667, abs=1e-5)
    assert lp.interval_mm_for_dpi(169) == pytest.approx(0.15029, abs=1e-4)
    with pytest.raises(ValueError):
        lp.interval_mm_for_dpi(0)


def test_identity_lut_and_apply() -> None:
    lp = _load_laser_physics()
    lut = lp.identity_lut()
    assert lut.shape == (256,)
    assert lut.dtype == np.uint8
    assert np.array_equal(lut, np.arange(256, dtype=np.uint8))
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    out = lp.apply_lut_to_gray(gray, lut)
    assert np.array_equal(out, gray)


def test_apply_lut_invalid_shape() -> None:
    lp = _load_laser_physics()
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        lp.apply_lut_to_gray(gray, np.arange(128, dtype=np.uint8))


def test_make_lut_callable_round_trips() -> None:
    """make_lut_callable produce un callable que aplica la LUT correctamente."""
    lp = _load_laser_physics()
    # LUT que invierte
    inv_lut = (255 - np.arange(256)).astype(np.uint8)
    f = lp.make_lut_callable(inv_lut)
    gray = np.array([[0, 50, 200, 255]], dtype=np.uint8)
    out = f(gray)
    expected = np.array([[255, 205, 55, 0]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_material_profile_acrylic_back_engrave() -> None:
    lp = _load_laser_physics()
    p = lp.acrylic_back_engrave_profile()
    assert p.name == "acrylic_back_engrave"
    assert p.spot_mm == pytest.approx(0.15)
    assert p.default_dpi == 169
    assert p.tone_response == "monotonic"
    # gamma 0.65 oscurece (aclara la entrada para compensar dot-gain): LUT[128] != 128
    assert p.lut_curve[128] != 128
    # LUT debe ser monotonica creciente para acrilico
    diffs = np.diff(p.lut_curve.astype(np.int64))
    assert (diffs >= 0).all(), "acrylic LUT debe ser monotonica creciente"
    assert p.max_useful_dpi() == 169


def test_material_profile_wood_non_monotonic_warning_friendly() -> None:
    lp = _load_laser_physics()
    p = lp.wood_profile()
    assert p.name == "wood_generic"
    assert p.tone_response == "non_monotonic"
    # LUT debe ser monotonica creciente (compresion del extremo) aunque la respuesta fisica
    # no lo sea: la LUT mapea pre-grabado, el rebote fisico se evita comprimiendo entrada.
    diffs = np.diff(p.lut_curve.astype(np.int64))
    assert (diffs >= 0).all()
    # Extremo claro comprimido: LUT[255] < 255
    assert p.lut_curve[255] < 255


def test_material_profile_validate_dpi() -> None:
    lp = _load_laser_physics()
    p = lp.acrylic_back_engrave_profile()
    assert p.validate_dpi(150, emit_warning=False) is None
    msg = p.validate_dpi(300, emit_warning=False)
    assert msg is not None and "300" in msg


def test_material_profile_lut_callable() -> None:
    lp = _load_laser_physics()
    p = lp.acrylic_back_engrave_profile()
    lut_fn = p.lut()
    gray = np.array([[128]], dtype=np.uint8)
    out = lut_fn(gray)
    assert out.shape == (1, 1)
    assert out.dtype == np.uint8
    assert out[0, 0] != 128


def test_load_material_profile_builtin() -> None:
    lp = _load_laser_physics()
    p = lp.load_material_profile("acrylic_back_engrave")
    assert p.name == "acrylic_back_engrave"


def test_load_material_profile_from_json(tmp_path: Path) -> None:
    lp = _load_laser_physics()
    payload = {
        "name": "custom_test",
        "spot_mm": 0.20,
        "default_dpi": 127,
        "lut_curve": list(range(256)),
        "tone_response": "linear",
        "power_pct_range": [10.0, 50.0],
        "notes": "test profile",
    }
    (tmp_path / "custom_test.json").write_text(json.dumps(payload), encoding="utf-8")
    p = lp.load_material_profile("custom_test", presets_dir=tmp_path)
    assert p.name == "custom_test"
    assert p.spot_mm == pytest.approx(0.20)
    assert p.default_dpi == 127
    assert p.tone_response == "linear"
    assert p.power_pct_range == (10.0, 50.0)
    assert np.array_equal(p.lut_curve, np.arange(256, dtype=np.uint8))


def test_load_material_profile_unknown_raises() -> None:
    lp = _load_laser_physics()
    with pytest.raises(KeyError, match="no encontrado"):
        lp.load_material_profile("nonexistent_material_xyz")


def test_load_material_profile_invalid_power_range() -> None:
    lp = _load_laser_physics()
    with pytest.raises(ValueError):
        lp.MaterialProfile(
            name="bad",
            spot_mm=0.15,
            default_dpi=169,
            lut_curve=lp.identity_lut(),
            power_pct_range=(80.0, 20.0),  # invertido
        )


def test_load_material_profile_npy_path(tmp_path: Path) -> None:
    """JSON puede referenciar `.npy` separado para la LUT (util para LUTs grandes calibradas)."""
    lp = _load_laser_physics()
    npy_path = tmp_path / "lut.npy"
    np.save(npy_path, np.arange(256, dtype=np.uint8))
    payload = {
        "name": "from_npy",
        "spot_mm": 0.15,
        "default_dpi": 169,
        "lut_curve_npy": "lut.npy",
        "tone_response": "monotonic",
        "power_pct_range": [5.0, 100.0],
    }
    (tmp_path / "from_npy.json").write_text(json.dumps(payload), encoding="utf-8")
    p = lp.load_material_profile("from_npy", presets_dir=tmp_path)
    assert np.array_equal(p.lut_curve, np.arange(256, dtype=np.uint8))


def test_scaled_unsharp_radius_for_typical_pipeline() -> None:
    """Ranking 240 px / output 100mm x 300dpi: radius 0.1mm fisico -> ~0.24 px ranking."""
    lp = _load_laser_physics()
    # output 100mm x 300dpi = 1181 px. ranking 240 -> scale = 240/1181 ≈ 0.2032
    # radius 0.1mm en output = 0.1 * 300/25.4 ≈ 1.18 px
    # radius en ranking = 1.18 * 0.2032 ≈ 0.24 px -> clamp 0.3
    r = lp.scaled_unsharp_radius(
        ranking_pixels_short_side=240,
        output_mm_short_side=100.0,
        output_dpi=300,
        radius_mm=0.10,
    )
    assert 0.3 <= r <= 0.5

    # Para output mas chico (50mm x 300dpi = 591 px), ranking 240 -> scale ~0.406
    # radius 0.1mm = 1.18 px output -> ranking 0.48 px
    r2 = lp.scaled_unsharp_radius(
        ranking_pixels_short_side=240,
        output_mm_short_side=50.0,
        output_dpi=300,
        radius_mm=0.10,
    )
    assert r2 > r  # ranking proporcionalmente mas grande del output -> radius mayor


def test_scaled_unsharp_radius_clamping() -> None:
    """Valores extremos quedan dentro de [0.3, 5.0]."""
    lp = _load_laser_physics()
    # radius_mm enorme -> clamp arriba
    r_high = lp.scaled_unsharp_radius(
        ranking_pixels_short_side=1000,
        output_mm_short_side=100.0,
        output_dpi=300,
        radius_mm=10.0,
    )
    assert r_high == 5.0
    # ranking minusculo -> clamp abajo
    r_low = lp.scaled_unsharp_radius(
        ranking_pixels_short_side=10,
        output_mm_short_side=200.0,
        output_dpi=300,
        radius_mm=0.05,
    )
    assert r_low == 0.3


def test_scaled_unsharp_radius_invalid_inputs() -> None:
    lp = _load_laser_physics()
    with pytest.raises(ValueError):
        lp.scaled_unsharp_radius(ranking_pixels_short_side=0, output_mm_short_side=100.0, output_dpi=300, radius_mm=0.1)
    with pytest.raises(ValueError):
        lp.scaled_unsharp_radius(ranking_pixels_short_side=240, output_mm_short_side=0.0, output_dpi=300, radius_mm=0.1)
    with pytest.raises(ValueError):
        lp.scaled_unsharp_radius(ranking_pixels_short_side=240, output_mm_short_side=100.0, output_dpi=0, radius_mm=0.1)
    with pytest.raises(ValueError):
        lp.scaled_unsharp_radius(ranking_pixels_short_side=240, output_mm_short_side=100.0, output_dpi=300, radius_mm=0.0)


def test_v5_with_acrylic_lut_changes_score() -> None:
    """Integracion: usar LUT del perfil acrilico cambia el score v5 vs identidad."""
    lp = _load_laser_physics()
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_scoring  # type: ignore[import-not-found]

    rng = np.random.default_rng(0)
    shape = (48, 48)
    gray = np.full(shape, 140, dtype=np.uint8)
    noise = rng.random(shape)
    out = np.where(noise >= 0.5, 255, 0).astype(np.uint8)

    profile = lp.acrylic_back_engrave_profile()
    score_no_lut, *_ = laser_scoring.score_candidate_v5(out, gray, candidate=None)
    score_with_lut, *_ = laser_scoring.score_candidate_v5(out, gray, candidate=None, lut=profile.lut())
    # La LUT acrilico aclara entradas (gamma 0.65) -> tone_local objetivo sube ->
    # menor desbalance vs out 50% blanco -> score deberia bajar (mejor).
    assert score_with_lut != pytest.approx(score_no_lut, abs=1e-6)
