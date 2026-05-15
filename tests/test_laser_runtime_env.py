"""Tests para helpers de entorno / GPU compartidos (sin corrida larga)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "laser_runtime_env.py"


def _load_runtime():
    spec = importlib.util.spec_from_file_location("laser_runtime_env", RUNTIME)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_torch_device_flag_cpu_cuda_gpu() -> None:
    m = _load_runtime()
    assert m.resolve_torch_device_flag("cpu") == "cpu"
    assert m.resolve_torch_device_flag("CPU") == "cpu"
    assert m.resolve_torch_device_flag("cuda") == "cuda"
    assert m.resolve_torch_device_flag("gpu") == "cuda"


def test_resolve_torch_device_flag_auto_is_cpu_or_cuda() -> None:
    m = _load_runtime()
    r = m.resolve_torch_device_flag("auto")
    assert r in ("cpu", "cuda")


def test_apply_cuda_process_memory_cap_smoke() -> None:
    m = _load_runtime()
    m.apply_cuda_process_memory_cap(quiet=True)


def test_apply_cuda_process_memory_cap_off_skips() -> None:
    m = _load_runtime()
    import os

    os.environ[m.ENV_CUDA_MEMORY_CAP_GIB] = "off"
    try:
        m.apply_cuda_process_memory_cap(quiet=True)
    finally:
        os.environ.pop(m.ENV_CUDA_MEMORY_CAP_GIB, None)


def test_infer_v4_max_gpu_workers_cap_sane_range() -> None:
    m = _load_runtime()
    c = m.infer_v4_max_gpu_workers_cap()
    assert isinstance(c, int)
    assert 1 <= c <= 8


def test_sync_hf_hub_token_env_aliases() -> None:
    import os

    m = _load_runtime()
    env: dict[str, str] = {}
    env[m.ENV_HF_TOKEN] = "hf_test_only"
    m.sync_hf_hub_token_env(env)
    assert env.get(m.ENV_HF_TOKEN_LEGACY) == "hf_test_only"

    env2: dict[str, str] = {}
    env2[m.ENV_HF_TOKEN_LEGACY] = "hf_legacy"
    m.sync_hf_hub_token_env(env2)
    assert env2.get(m.ENV_HF_TOKEN) == "hf_legacy"


def test_child_process_env_preserves_hf_token() -> None:
    import os

    m = _load_runtime()
    old_hf = os.environ.pop(m.ENV_HF_TOKEN, None)
    old_hub = os.environ.pop(m.ENV_HF_TOKEN_LEGACY, None)
    try:
        os.environ[m.ENV_HF_TOKEN] = "hf_from_parent"
        child = m.child_process_env()
        assert child.get(m.ENV_HF_TOKEN) == "hf_from_parent"
        assert child.get(m.ENV_HF_TOKEN_LEGACY) == "hf_from_parent"
    finally:
        if old_hf is not None:
            os.environ[m.ENV_HF_TOKEN] = old_hf
        else:
            os.environ.pop(m.ENV_HF_TOKEN, None)
        if old_hub is not None:
            os.environ[m.ENV_HF_TOKEN_LEGACY] = old_hub
        else:
            os.environ.pop(m.ENV_HF_TOKEN_LEGACY, None)
