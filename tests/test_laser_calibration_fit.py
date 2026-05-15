"""Tests `laser_calibration_fit.py`: ajuste de LUT desde foto sintetica del wedge."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


def _make_wedge_outputs(tmp_path: Path) -> tuple[Path, dict]:
    """Genera un wedge real en tmp_path; devuelve ruta PNG y meta dict."""
    lcw = _load("laser_calibration_wedge")
    patches, w, h = lcw.layout_patches(
        steps=8, square_mm=10.0, gap_mm=2.0, margin_mm=3.0, dpi=169
    )
    canvas = lcw.render_wedge(patches, w, h, label_each=False, dpi=169)
    out_png = tmp_path / "wedge.png"
    meta = lcw.save_wedge_outputs(canvas, patches, out_png, dpi=169, material=None)
    return out_png, meta


def _synthesize_engraved_photo(
    wedge_path: Path,
    meta: dict,
    response_gamma: float,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Sintetiza una "foto del wedge grabado": para cada parche, reemplaza el dither
    binario por un gris constante `255 * (input_gray/255)**response_gamma`.

    response_gamma > 1 oscurece (dot-gain tipico de laser); < 1 aclara.
    """
    rng = np.random.default_rng(seed)
    photo = np.array(Image.open(wedge_path), dtype=np.float64)
    for p in meta["patches"]:
        x, y, w, h = int(p["x_px"]), int(p["y_px"]), int(p["w_px"]), int(p["h_px"])
        input_norm = float(p["input_gray"]) / 255.0
        measured = 255.0 * (input_norm ** response_gamma)
        photo[y:y + h, x:x + w] = measured
    if noise_sigma > 0:
        photo += rng.normal(0.0, noise_sigma, size=photo.shape)
    return np.clip(photo, 0, 255).astype(np.uint8)


def test_load_wedge_meta_rejects_wrong_schema(tmp_path: Path) -> None:
    lcf = _load("laser_calibration_fit")
    bad = tmp_path / "bad_meta.json"
    bad.write_text(json.dumps({"schema": "something_else", "patches": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        lcf.load_wedge_meta(bad)


def test_measure_patches_extracts_means(tmp_path: Path) -> None:
    lcf = _load("laser_calibration_fit")
    wedge_path, meta = _make_wedge_outputs(tmp_path)
    photo_arr = _synthesize_engraved_photo(wedge_path, meta, response_gamma=1.0, noise_sigma=0)
    photo_jpg = tmp_path / "photo.png"
    Image.fromarray(photo_arr, mode="L").save(photo_jpg)
    photo = lcf.load_photo_as_gray(photo_jpg)
    photo = lcf.resize_to_wedge(photo, tuple(meta["image_size_px"]))
    measurements = lcf.measure_patches(photo, meta)
    assert len(measurements) == meta["n_patches"]
    # Con gamma 1.0 measured ≈ input_gray
    for m in measurements:
        assert abs(m.measured_gray - m.input_gray) < 5.0, (
            f"patch {m.index} esperaba ~{m.input_gray} got {m.measured_gray:.1f}"
        )


def test_fit_recovers_inverse_gamma(tmp_path: Path) -> None:
    """Material 'oscurecedor' (gamma 2.0) debe producir LUT que aclare (gamma ~0.5)."""
    lcf = _load("laser_calibration_fit")
    wedge_path, meta = _make_wedge_outputs(tmp_path)
    photo_arr = _synthesize_engraved_photo(wedge_path, meta, response_gamma=2.0, noise_sigma=0.5)
    photo_jpg = tmp_path / "photo.png"
    Image.fromarray(photo_arr, mode="L").save(photo_jpg)
    photo = lcf.load_photo_as_gray(photo_jpg)
    photo = lcf.resize_to_wedge(photo, tuple(meta["image_size_px"]))
    measurements = lcf.measure_patches(photo, meta)
    lut, debug = lcf.fit_inverse_lut(measurements)
    assert lut.shape == (256,)
    assert lut.dtype == np.uint8
    # La LUT debe aclarar entradas (compensar el oscurecimiento del material).
    # Para input=128 (medio), el material lo grabaria como ~57 (128*0.5^1=...).
    # Concretamente con gamma=2, input pre-LUT 181 -> medido 128.
    # Asi LUT[128] (lo que pre-cargamos para que se vea como 128) debe ser ~181.
    expected_lut_128 = round(255.0 * ((128.0 / 255.0) ** (1.0 / 2.0)))  # ~181
    actual_lut_128 = int(lut[128])
    assert abs(actual_lut_128 - expected_lut_128) < 15, (
        f"LUT[128]={actual_lut_128} lejos del esperado {expected_lut_128} para gamma=2.0"
    )


def test_fit_identity_for_linear_response(tmp_path: Path) -> None:
    """Material lineal (gamma 1.0) debe producir LUT ~identidad en el rango medido."""
    lcf = _load("laser_calibration_fit")
    wedge_path, meta = _make_wedge_outputs(tmp_path)
    photo_arr = _synthesize_engraved_photo(wedge_path, meta, response_gamma=1.0, noise_sigma=0)
    photo_jpg = tmp_path / "photo.png"
    Image.fromarray(photo_arr, mode="L").save(photo_jpg)
    photo = lcf.load_photo_as_gray(photo_jpg)
    photo = lcf.resize_to_wedge(photo, tuple(meta["image_size_px"]))
    measurements = lcf.measure_patches(photo, meta)
    lut, _ = lcf.fit_inverse_lut(measurements)
    # En el rango medido (8 parches de 0..255), LUT[g] ≈ g con tolerancia
    for g in (40, 100, 160, 220):
        assert abs(int(lut[g]) - g) < 12, f"LUT[{g}]={int(lut[g])} no es identidad"


def test_fit_handles_non_monotonic_with_isotonic(tmp_path: Path) -> None:
    """Si las mediciones no son monotonicas (madera), isotonic regression actua."""
    lcf = _load("laser_calibration_fit")
    measurements = [
        lcf.PatchMeasurement(index=0, input_gray=0, measured_gray=10.0, pixel_count=10, center_px=(0, 0)),
        lcf.PatchMeasurement(index=1, input_gray=64, measured_gray=80.0, pixel_count=10, center_px=(0, 0)),
        lcf.PatchMeasurement(index=2, input_gray=128, measured_gray=140.0, pixel_count=10, center_px=(0, 0)),
        # rebote: input 192 mide MENOS que input 128 (lignina sublima -> aclara)
        lcf.PatchMeasurement(index=3, input_gray=192, measured_gray=110.0, pixel_count=10, center_px=(0, 0)),
        lcf.PatchMeasurement(index=4, input_gray=255, measured_gray=130.0, pixel_count=10, center_px=(0, 0)),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lut, debug = lcf.fit_inverse_lut(measurements, force_monotonic=True)
    assert debug["non_monotonic_violations_input"] > 0
    assert any("monotonic" in str(w.message).lower() for w in caught)
    # La LUT debe seguir siendo razonable (uint8, monotonica creciente o casi)
    assert lut.dtype == np.uint8
    diffs = np.diff(lut.astype(np.int64))
    # tolerar pequenas oscilaciones por interpolacion en regiones planas
    assert (diffs >= -1).all(), "LUT debe ser practicamente monotonica creciente tras isotonic"


def test_save_lut_creates_npy_and_sidecar(tmp_path: Path) -> None:
    lcf = _load("laser_calibration_fit")
    lut = np.arange(256, dtype=np.uint8)
    out_npy = tmp_path / "lut_test.npy"
    payload = lcf.save_lut(lut, {"foo": "bar"}, out_npy, material_name="test_mat")
    assert out_npy.is_file()
    sidecar = out_npy.with_suffix(".json")
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["schema"] == "laser_calibration_fit/v1"
    assert data["lut_npy"] == "lut_test.npy"
    assert data["material"] == "test_mat"
    loaded = np.load(out_npy)
    assert np.array_equal(loaded, lut)


def test_integration_lut_usable_in_material_profile(tmp_path: Path) -> None:
    """Cierra el loop: el LUT producido se carga via laser_physics.MaterialProfile."""
    lcf = _load("laser_calibration_fit")
    lp = _load("laser_physics")
    wedge_path, meta = _make_wedge_outputs(tmp_path)
    photo_arr = _synthesize_engraved_photo(wedge_path, meta, response_gamma=1.5, noise_sigma=0.5)
    photo_jpg = tmp_path / "photo.png"
    Image.fromarray(photo_arr, mode="L").save(photo_jpg)
    photo = lcf.load_photo_as_gray(photo_jpg)
    photo = lcf.resize_to_wedge(photo, tuple(meta["image_size_px"]))
    measurements = lcf.measure_patches(photo, meta)
    lut, _debug = lcf.fit_inverse_lut(measurements)
    # Guardamos como JSON con lut inline para que laser_physics.load_material_profile lo cargue
    payload = {
        "name": "fitted_acrylic",
        "spot_mm": 0.15,
        "default_dpi": 169,
        "lut_curve": lut.tolist(),
        "tone_response": "monotonic",
        "power_pct_range": [9.0, 14.0],
        "notes": "Sintetico via test_integration",
    }
    (tmp_path / "fitted_acrylic.json").write_text(json.dumps(payload), encoding="utf-8")
    profile = lp.load_material_profile("fitted_acrylic", presets_dir=tmp_path)
    assert profile.name == "fitted_acrylic"
    assert profile.lut_curve.shape == (256,)
    # Aplicar la LUT a gris 128 debe oscurecerlo (porque material gamma 1.5 oscurece y la LUT compensa subiendo entrada)
    # Mas robusto: solo verificar que la LUT no es identidad
    assert not np.array_equal(profile.lut_curve, np.arange(256, dtype=np.uint8))
