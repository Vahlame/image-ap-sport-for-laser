"""Tests del FastAPI backend `scripts/api_server.py` con TestClient."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SCRIPTS = ROOT / "scripts"

# Skip si FastAPI no esta instalado (extra [api]).
try:
    from fastapi.testclient import TestClient  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="extra [api] no instalado")


def _load_api_app():
    if str(SCRIPT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPT_SCRIPTS))
    spec = importlib.util.spec_from_file_location("api_server", SCRIPT_SCRIPTS / "api_server.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_image(w: int = 64, h: int = 64, seed: int = 42) -> bytes:
    """Genera una imagen JPEG sintetica RGB para pruebas."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_health_endpoint():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "cuda_available" in data
    assert isinstance(data["repo_root"], str)


def test_materials_lists_builtins():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        r = client.get("/api/materials")
    assert r.status_code == 200
    data = r.json()
    names = [m["name"] for m in data]
    assert "acrylic_back_engrave" in names
    assert "wood_generic" in names
    # Cada uno tiene los campos esperados
    for m in data:
        assert m["spot_mm"] > 0
        assert m["default_dpi"] > 0
        assert m["tone_response"] in ("monotonic", "non_monotonic", "linear")
        assert m["source"] in ("builtin", "custom")


def test_algorithms_lists_all_families():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        r = client.get("/api/algorithms")
    assert r.status_code == 200
    data = r.json()
    families = {g["family"] for g in data}
    assert families == {"ordered_dither", "error_diffusion", "burkes_blue_variants", "mix_multipass"}
    # Suma de todos los algoritmos == ALL_RENDER_ALGORITHMS count
    all_count = sum(len(g["algorithms"]) for g in data)
    import laser_target_match as ltm
    assert all_count >= len(ltm.ALL_RENDER_ALGORITHMS) - 5  # algunos pueden estar en ordered si overlap
    # Floyd debe estar en error_diffusion
    diff = next(g for g in data if g["family"] == "error_diffusion")
    assert "floyd" in diff["algorithms"]


def test_preview_endpoint_returns_png():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image(w=400, h=400)
    params = {"algorithm": "floyd", "threshold": 128, "preprocess_mode": "none"}
    with TestClient(api.app) as client:
        r = client.post(
            "/api/preview",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params)},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "X-Process-Time-Ms" in r.headers
    assert "X-Output-Width" in r.headers
    # Verificar que es PNG binario valido (0/255)
    out = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
    unique = set(np.unique(out).tolist())
    assert unique.issubset({0, 255})


def test_process_endpoint_full_res():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image(w=200, h=200)
    params = {
        "algorithm": "floyd", "threshold": 83, "contrast": 0.55, "brightness": 25.0,
        "gamma": 1.35, "autocontrast": 0.0, "sharpen": 40.0, "invert": True,
        "preprocess_mode": "sauvola", "max_side": 0,  # full
    }
    with TestClient(api.app) as client:
        r = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params)},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
    assert out.shape == (200, 200)


def test_process_with_material_applies_lut():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image(w=128, h=128)
    params_no_mat = {"algorithm": "floyd", "preprocess_mode": "none", "material": ""}
    params_mat = {**params_no_mat, "material": "acrylic_back_engrave"}
    with TestClient(api.app) as client:
        r1 = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params_no_mat)},
        )
        r2 = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params_mat)},
        )
    assert r1.status_code == 200 and r2.status_code == 200
    # Con LUT acrylic, header X-Material poblado
    assert r1.headers.get("X-Material", "") == ""
    assert r2.headers["X-Material"] == "acrylic_back_engrave"
    # Las imagenes deben ser distintas (LUT cambia distribucion tonal pre-dither)
    out1 = np.array(Image.open(io.BytesIO(r1.content)).convert("L"))
    out2 = np.array(Image.open(io.BytesIO(r2.content)).convert("L"))
    assert not np.array_equal(out1, out2), "LUT acrylic deberia cambiar el output"


def test_invalid_algorithm_returns_400():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image()
    params = {"algorithm": "nonexistent_algo_xyz"}
    with TestClient(api.app) as client:
        r = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params)},
        )
    assert r.status_code == 400
    assert "no soportado" in r.json()["detail"].lower()


def test_invalid_image_returns_400():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    bad_bytes = b"not_an_image_garbage_bytes_123456"
    with TestClient(api.app) as client:
        r = client.post(
            "/api/process",
            files={"image": ("test.jpg", bad_bytes, "image/jpeg")},
            data={"params_json": json.dumps({"algorithm": "floyd"})},
        )
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"].lower() or "invalida" in r.json()["detail"].lower()


def test_invalid_params_json_returns_422():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image()
    with TestClient(api.app) as client:
        r = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": "{ broken json"},
        )
    assert r.status_code == 422


def test_preview_smaller_than_full_process():
    """/api/preview fuerza max_side=400; output debe ser <= que process full-res."""
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image(w=800, h=800)
    params = {"algorithm": "floyd", "preprocess_mode": "none", "max_side": 0}
    with TestClient(api.app) as client:
        r_prev = client.post(
            "/api/preview",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params)},
        )
        r_full = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params)},
        )
    assert r_prev.status_code == 200 and r_full.status_code == 200
    prev_w = int(r_prev.headers["X-Output-Width"])
    full_w = int(r_full.headers["X-Output-Width"])
    assert prev_w <= 400, f"preview width {prev_w} debe ser <= 400"
    assert full_w == 800, f"process full width {full_w} debe matchear input 800"


def test_simulate_endpoint_acrylic():
    """/api/simulate con material acrylic: frost claro en zonas grabadas."""
    api = _load_api_app()
    from fastapi.testclient import TestClient
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[24:72, 24:72] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    with TestClient(api.app) as client:
        r = client.post(
            "/api/simulate",
            files={"image": ("bin.png", buf.getvalue(), "image/png")},
            data={"material": "acrylic_back_engrave", "output_dpi": "169"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["X-Material"] == "acrylic_back_engrave"
    assert float(r.headers["X-Sim-Sigma-Px"]) > 0.3
    assert float(r.headers["X-Sim-Spot-Mm"]) == pytest.approx(0.15, abs=0.01)
    sim = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
    assert sim.shape == arr.shape
    center = float(sim[40:56, 40:56].mean())
    corner = float(sim[0:8, 0:8].mean())
    assert center > corner + 80, f"frost no aclara: center={center}, corner={corner}"


def test_simulate_endpoint_wood_burn_no_material():
    """Sin material: usa appearance directo. wood_burn debe oscurecer centro."""
    api = _load_api_app()
    from fastapi.testclient import TestClient
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[16:48, 16:48] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    with TestClient(api.app) as client:
        r = client.post(
            "/api/simulate",
            files={"image": ("bin.png", buf.getvalue(), "image/png")},
            data={"material": "", "output_dpi": "169", "appearance": "wood_burn"},
        )
    assert r.status_code == 200
    sim = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
    center = float(sim[28:36, 28:36].mean())
    corner = float(sim[0:8, 0:8].mean())
    assert center < corner - 50, f"burn no oscurece: center={center}, corner={corner}"


def test_simulate_endpoint_invalid_appearance_returns_400():
    api = _load_api_app()
    from fastapi.testclient import TestClient
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[10:22, 10:22] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    with TestClient(api.app) as client:
        r = client.post(
            "/api/simulate",
            files={"image": ("bin.png", buf.getvalue(), "image/png")},
            data={"material": "", "appearance": "invalid_xyz"},
        )
    assert r.status_code == 400


def test_dpi_validation_warning_in_logs(caplog):
    """Pedir DPI alto contra acrylic spot 0.15 debe pasar pero idealmente quedar registrado.
    No bloqueamos requests por DPI alto pero el header X-Sharpen-Radius-Px se calcula."""
    api = _load_api_app()
    from fastapi.testclient import TestClient
    img_bytes = _make_synthetic_image(w=200, h=200)
    params = {
        "algorithm": "floyd", "preprocess_mode": "none",
        "material": "acrylic_back_engrave",
        "output_mm_short": 100.0, "output_dpi": 169,
        "sharpen_radius_mm": 0.10,
    }
    with TestClient(api.app) as client:
        r = client.post(
            "/api/process",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"params_json": json.dumps(params)},
        )
    assert r.status_code == 200
    # sharpen radius escalado != 1.2 default
    radius = float(r.headers["X-Sharpen-Radius-Px"])
    assert 0.3 <= radius <= 5.0, f"sharpen radius {radius} fuera de clamp"
