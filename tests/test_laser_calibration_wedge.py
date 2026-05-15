"""Tests para `scripts/laser_calibration_wedge.py`: generador de step-wedge calibracion."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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


def test_step_gray_values_linear() -> None:
    lcw = _load("laser_calibration_wedge")
    g = lcw.step_gray_values(5, gamma=1.0)
    assert g == [0, 64, 128, 191, 255]
    # 16 pasos cubren rango completo monotonicamente
    g16 = lcw.step_gray_values(16)
    assert g16[0] == 0 and g16[-1] == 255
    diffs = np.diff(g16)
    assert (diffs >= 0).all()


def test_step_gray_values_invalid_steps() -> None:
    lcw = _load("laser_calibration_wedge")
    with pytest.raises(ValueError):
        lcw.step_gray_values(1)


def test_step_gray_values_gamma_changes_distribution() -> None:
    """Gamma > 1 concentra valores en oscuros (util para acrilico bajo potencia)."""
    lcw = _load("laser_calibration_wedge")
    g_lin = lcw.step_gray_values(8, gamma=1.0)
    g_dark = lcw.step_gray_values(8, gamma=2.0)
    # En gamma > 1, los pasos intermedios estan mas cerca de 0 que en lineal
    assert g_dark[4] < g_lin[4]


def test_layout_patches_grid() -> None:
    lcw = _load("laser_calibration_wedge")
    patches, w, h = lcw.layout_patches(
        steps=16, square_mm=10.0, gap_mm=3.0, margin_mm=5.0, dpi=169
    )
    assert len(patches) == 16
    # 4x4 grid (sqrt(16))
    assert max(p.x_px for p in patches) > 0
    # No solapamiento de parches
    for i, p1 in enumerate(patches):
        for p2 in patches[i + 1:]:
            xi, yi = p1.x_px, p1.y_px
            wi, hi = p1.w_px, p1.h_px
            xj, yj = p2.x_px, p2.y_px
            wj, hj = p2.w_px, p2.h_px
            overlap = not (xi + wi <= xj or xj + wj <= xi or yi + hi <= yj or yj + hj <= yi)
            assert not overlap, f"patch {p1.index} solapa con {p2.index}"
    assert w > 0 and h > 0


def test_layout_patches_explicit_cols() -> None:
    lcw = _load("laser_calibration_wedge")
    patches, _, _ = lcw.layout_patches(
        steps=12, square_mm=8.0, gap_mm=2.0, margin_mm=4.0, dpi=200, cols=6
    )
    # 12 parches en 6 cols -> 2 rows
    rows_seen = set()
    for p in patches:
        rows_seen.add(p.y_px)
    assert len(rows_seen) == 2


def test_render_patch_floyd_produces_dithered_binary() -> None:
    """Floyd debe convertir gris constante en patron binario con white_ratio cercano al gris/255."""
    lcw = _load("laser_calibration_wedge")
    p = lcw.PatchSpec(
        index=0, input_gray=128, x_px=0, y_px=0, w_px=32, h_px=32, dither="floyd", label="128"
    )
    out = lcw.render_patch(p)
    assert out.shape == (32, 32)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 255})
    wr = float((out == 255).mean())
    # Floyd con threshold=128 sobre gris=128: aproximacion balanceada; tolerancia amplia
    assert 0.3 <= wr <= 0.7


def test_render_patch_bayer8() -> None:
    lcw = _load("laser_calibration_wedge")
    p = lcw.PatchSpec(
        index=0, input_gray=64, x_px=0, y_px=0, w_px=24, h_px=24, dither="bayer8", label="64"
    )
    out = lcw.render_patch(p)
    wr = float((out == 255).mean())
    # gris 64 -> white_ratio ~ 64/255 ~ 0.25; tolerancia amplia
    assert 0.10 <= wr <= 0.40


def test_render_patch_blue_noise_vac32() -> None:
    """Verifica que el dither blue-noise voi-and-cluster integra bien."""
    lcw = _load("laser_calibration_wedge")
    p = lcw.PatchSpec(
        index=0, input_gray=140, x_px=0, y_px=0, w_px=64, h_px=64, dither="blue_noise_vac32", label="140"
    )
    out = lcw.render_patch(p)
    wr = float((out == 255).mean())
    # gris 140 -> white_ratio ~ 140/255 ~ 0.55; tolerancia amplia
    assert 0.40 <= wr <= 0.70


def test_render_patch_threshold() -> None:
    lcw = _load("laser_calibration_wedge")
    p1 = lcw.PatchSpec(index=0, input_gray=100, x_px=0, y_px=0, w_px=8, h_px=8, dither="threshold", label="100")
    p2 = lcw.PatchSpec(index=1, input_gray=200, x_px=0, y_px=0, w_px=8, h_px=8, dither="threshold", label="200")
    out1 = lcw.render_patch(p1)
    out2 = lcw.render_patch(p2)
    assert (out1 == 0).all()
    assert (out2 == 255).all()


def test_render_patch_unknown_dither_raises() -> None:
    lcw = _load("laser_calibration_wedge")
    p = lcw.PatchSpec(index=0, input_gray=128, x_px=0, y_px=0, w_px=8, h_px=8, dither="nonexistent", label="x")
    with pytest.raises(ValueError):
        lcw.render_patch(p)


def test_save_wedge_outputs_creates_png_and_meta(tmp_path: Path) -> None:
    lcw = _load("laser_calibration_wedge")
    patches, w, h = lcw.layout_patches(
        steps=4, square_mm=5.0, gap_mm=1.0, margin_mm=2.0, dpi=169
    )
    canvas = lcw.render_wedge(patches, w, h, label_each=False, dpi=169)
    out_png = tmp_path / "wedge.png"
    meta = lcw.save_wedge_outputs(canvas, patches, out_png, dpi=169, material="test_material")
    assert out_png.is_file()
    meta_path = tmp_path / "wedge_meta.json"
    assert meta_path.is_file()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["schema"] == "laser_calibration_wedge/v1"
    assert data["n_patches"] == 4
    assert data["material"] == "test_material"
    assert data["dpi"] == 169
    # PNG legible
    img = Image.open(out_png)
    assert img.mode == "L"
    arr = np.array(img)
    assert arr.shape == (h, w)


def test_cli_smoke(tmp_path: Path) -> None:
    """CLI con argumentos minimos produce PNG + meta."""
    out_png = tmp_path / "cli_wedge.png"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_SCRIPTS / "laser_calibration_wedge.py"),
            "--out", str(out_png),
            "--steps", "9",
            "--square-mm", "8.0",
            "--gap-mm", "2.0",
            "--margin-mm", "3.0",
            "--dpi", "169",
            "--dither", "bayer8",
            "--no-labels",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"CLI exit nonzero: {result.stderr}"
    assert out_png.is_file()
    meta_path = tmp_path / "cli_wedge_meta.json"
    assert meta_path.is_file()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["n_patches"] == 9
    assert data["dpi"] == 169


def test_cli_with_material_validates_dpi(tmp_path: Path) -> None:
    """CLI con --material acrilico y DPI=300 (>169) debe avisar pero no fallar."""
    out_png = tmp_path / "wedge_acrylic.png"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_SCRIPTS / "laser_calibration_wedge.py"),
            "--out", str(out_png),
            "--steps", "4",
            "--square-mm", "8.0",
            "--gap-mm", "2.0",
            "--margin-mm", "3.0",
            "--dpi", "300",  # excede 1/spot(0.15) = 169
            "--dither", "floyd",
            "--no-labels",
            "--material", "acrylic_back_engrave",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert out_png.is_file()
    # Mensaje de warning debe aparecer en stdout
    combined = result.stdout + result.stderr
    assert "MATERIAL" in combined
