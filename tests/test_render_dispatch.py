"""Tests del nuevo dispatch de render_candidate (R6 refactor).

Verifica:
- ALL_RENDER_ALGORITHMS cubre las 3 tablas sin overlap.
- Cada algoritmo en cada tabla produce uint8 binario {0, 255} sin levantar.
- Algoritmo desconocido levanta ValueError con mensaje legible.
- RESTART_ALGORITHMS solo contiene nombres validos (proteccion contra drift).
"""

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


def test_all_render_algorithms_is_disjoint_union() -> None:
    """Las 3 tablas (DIFFUSION, BURKES_BLUE, NAMED) no deben solaparse en nombres."""
    ltm = _load("laser_target_match")
    keys_diff = set(ltm.DIFFUSION_ALGORITHMS.keys())
    keys_burkes = set(ltm.BURKES_BLUE_VARIANTS.keys())
    keys_named = set(ltm.NAMED_RENDERERS.keys())
    assert not (keys_diff & keys_burkes), f"overlap DIFFUSION/BURKES: {keys_diff & keys_burkes}"
    assert not (keys_diff & keys_named), f"overlap DIFFUSION/NAMED: {keys_diff & keys_named}"
    assert not (keys_burkes & keys_named), f"overlap BURKES/NAMED: {keys_burkes & keys_named}"
    assert set(ltm.ALL_RENDER_ALGORITHMS) == keys_diff | keys_burkes | keys_named


def test_all_render_algorithms_count() -> None:
    """Garantiza que el inventario no se rompe silenciosamente."""
    ltm = _load("laser_target_match")
    # Si baja: confirmar intencion (puede ser limpieza valida).
    # Si sube: bienvenido pero verificar test_all_render_algorithms_smoke ejercita el nuevo.
    assert len(ltm.ALL_RENDER_ALGORITHMS) >= 40, (
        f"esperaba >= 40 algoritmos, hay {len(ltm.ALL_RENDER_ALGORITHMS)}"
    )


def test_render_candidate_smoke_all_algorithms() -> None:
    """Cada algoritmo de las 3 tablas produce uint8 binario sin levantar."""
    ltm = _load("laser_target_match")
    rng = np.random.default_rng(42)
    gray = rng.integers(40, 200, size=(48, 48), dtype=np.uint8).astype(np.float64)

    failures: list[str] = []
    for algo in ltm.ALL_RENDER_ALGORITHMS:
        try:
            cand = ltm.Candidate(
                algorithm=algo, invert=False, threshold=128,
                contrast=1.0, brightness=0.0, gamma=1.0, autocontrast=0.0, sharpen=0.0,
            )
            out = ltm.render_candidate(gray, cand)
            if out.dtype != np.uint8:
                failures.append(f"{algo}: dtype esperado uint8 got {out.dtype}")
            if out.shape != gray.shape:
                failures.append(f"{algo}: shape esperado {gray.shape} got {out.shape}")
            unique = set(np.unique(out).tolist())
            if not unique.issubset({0, 255}):
                failures.append(f"{algo}: valores fuera de 0/255: {unique}")
        except Exception as exc:
            failures.append(f"{algo}: EXCEPTION {type(exc).__name__}: {exc}")

    assert not failures, "Algoritmos rotos tras refactor:\n  " + "\n  ".join(failures)


def test_render_candidate_unknown_algorithm_raises() -> None:
    ltm = _load("laser_target_match")
    gray = np.full((16, 16), 128, dtype=np.float64)
    cand = ltm.Candidate(
        algorithm="nonexistent_xyz", invert=False, threshold=128,
        contrast=1.0, brightness=0.0, gamma=1.0, autocontrast=0.0, sharpen=0.0,
    )
    with pytest.raises(ValueError, match="no soportado"):
        ltm.render_candidate(gray, cand)


def test_restart_algorithms_subset_of_all() -> None:
    """RESTART_ALGORITHMS no debe contener nombres que el dispatch no reconozca."""
    ltm = _load("laser_target_match")
    invalid = [a for a in ltm.RESTART_ALGORITHMS if a not in ltm.ALL_RENDER_ALGORITHMS]
    assert not invalid, f"RESTART_ALGORITHMS tiene nombres no soportados: {invalid}"


def test_brutal_restart_algorithms_subset_of_all() -> None:
    """BRUTAL_RESTART_ALGORITHMS idem (extension de RESTART_ALGORITHMS)."""
    ltm = _load("laser_target_match")
    invalid = [a for a in ltm.BRUTAL_RESTART_ALGORITHMS if a not in ltm.ALL_RENDER_ALGORITHMS]
    assert not invalid, f"BRUTAL_RESTART_ALGORITHMS tiene nombres no soportados: {invalid}"


def test_burkes_blue_variants_structure() -> None:
    """Cada entry debe ser (mid_lo, mid_hi, blue_strength)."""
    ltm = _load("laser_target_match")
    for name, value in ltm.BURKES_BLUE_VARIANTS.items():
        assert len(value) == 3, f"{name}: esperaba (mid_lo, mid_hi, blue_strength)"
        mid_lo, mid_hi, blue_strength = value
        assert 0 <= mid_lo <= mid_hi <= 255, f"{name}: rango midtone invalido {mid_lo}..{mid_hi}"
        assert 0 < blue_strength <= 200, f"{name}: blue_strength fuera de rango {blue_strength}"


def test_named_renderers_callable_signature() -> None:
    """Cada renderer debe aceptar (gray, candidate) y devolver uint8 binario."""
    ltm = _load("laser_target_match")
    gray = np.full((24, 24), 130, dtype=np.float64)
    cand = ltm.Candidate(
        algorithm="threshold", invert=False, threshold=128,
        contrast=1.0, brightness=0.0, gamma=1.0, autocontrast=0.0, sharpen=0.0,
    )
    for name, fn in ltm.NAMED_RENDERERS.items():
        out = fn(gray, cand)
        assert isinstance(out, np.ndarray), f"{name}: salida no es ndarray"
        assert out.dtype == np.uint8, f"{name}: dtype {out.dtype}"
        assert out.shape == gray.shape, f"{name}: shape {out.shape}"
