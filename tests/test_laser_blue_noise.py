"""Tests para `scripts/laser_blue_noise.py`: void-and-cluster (Ulichney 1993).

Verifica:
- Generacion produce permutacion completa [0, n).
- Cache funciona (segunda llamada usa disco).
- Espectro del binario obtenido al threshold 0.5 tiene poca energia en baja
  frecuencia (criterio blue-noise) — mejor que random/cluster.
"""

from __future__ import annotations

import importlib.util
import sys
import time
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


def test_void_and_cluster_returns_permutation_small() -> None:
    """size=8 (rapido): valores deben ser permutacion completa de [0, 64)."""
    lbn = _load("laser_blue_noise")
    m = lbn.generate_void_and_cluster(size=8, seed=123)
    assert m.shape == (8, 8)
    assert m.dtype == np.int32
    assert sorted(m.flatten().tolist()) == list(range(64))


def test_threshold_matrix_in_range() -> None:
    """threshold matrix tiene valores estrictamente en (0, 1)."""
    lbn = _load("laser_blue_noise")
    t = lbn.threshold_matrix_for_dithering(size=8)
    assert t.shape == (8, 8)
    assert (t > 0.0).all() and (t < 1.0).all()


def test_cache_persists_and_reuses(tmp_path: Path) -> None:
    """Generado se persiste; segunda llamada usa el .npy."""
    lbn = _load("laser_blue_noise")
    t0 = time.perf_counter()
    m1 = lbn.void_and_cluster_matrix(size=8, cache_dir=tmp_path, seed=777)
    elapsed_gen = time.perf_counter() - t0
    cache_file = tmp_path / "blue_noise_8.npy"
    assert cache_file.is_file()
    t1 = time.perf_counter()
    m2 = lbn.void_and_cluster_matrix(size=8, cache_dir=tmp_path)
    elapsed_load = time.perf_counter() - t1
    assert np.array_equal(m1, m2)
    # disco debe ser al menos un orden de magnitud mas rapido (tolerante)
    assert elapsed_load < max(0.05, elapsed_gen / 2.0), (
        f"cache no acelero (gen={elapsed_gen:.3f}s, load={elapsed_load:.3f}s)"
    )


def test_cache_force_regen(tmp_path: Path) -> None:
    """force_regen ignora el cache existente."""
    lbn = _load("laser_blue_noise")
    m1 = lbn.void_and_cluster_matrix(size=8, cache_dir=tmp_path, seed=42)
    # cambiar la seed regenera con force_regen
    m2 = lbn.void_and_cluster_matrix(size=8, cache_dir=tmp_path, force_regen=True, seed=999)
    assert not np.array_equal(m1, m2)


def _spectral_lowfreq_ratio(binary: np.ndarray, low_band_fraction: float = 0.10) -> float:
    """Replica simplificada de la metrica para validar perfil espectral."""
    img = binary.astype(np.float64) / 255.0
    img = img - img.mean()
    h, w = img.shape
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    radial_sum = np.bincount(r.ravel(), weights=power.ravel())
    radial_count = np.bincount(r.ravel())
    radial_avg = radial_sum / np.maximum(radial_count, 1)
    radial_avg[0] = 0.0
    n_bins = len(radial_avg)
    cutoff = max(1, int(round(n_bins * float(low_band_fraction))))
    low = float(radial_avg[1:cutoff + 1].mean())
    total = float(radial_avg[1:].mean()) + 1e-12
    return low / (low + total)


def test_blue_noise_better_lowfreq_than_random_threshold() -> None:
    """
    Al umbralizar un gris constante 128 con la matriz blue-noise, el binario
    resultante debe tener menos energia en baja frecuencia que con random.
    Esta es la propiedad fundamental de blue-noise.
    """
    lbn = _load("laser_blue_noise")
    size = 16
    t_bn = lbn.threshold_matrix_for_dithering(size=size, seed=12345)
    # tilear a 96x96 y umbralizar gris=128
    gray = np.full((96, 96), 128, dtype=np.uint8) / 255.0
    tile = np.tile(t_bn, (96 // size + 1, 96 // size + 1))[:96, :96]
    out_bn = np.where(gray >= tile, 255, 0).astype(np.uint8)

    # comparativa: random thresholds uniformes
    rng = np.random.default_rng(0)
    tile_random = rng.random((96, 96))
    out_random = np.where(gray >= tile_random, 255, 0).astype(np.uint8)

    lf_bn = _spectral_lowfreq_ratio(out_bn)
    lf_random = _spectral_lowfreq_ratio(out_random)
    # Random (white-noise spectrum plano) tiende a ~0.5 (energia equitativa entre bandas).
    # Blue-noise debe quedar claramente por debajo: la propiedad fundamental es atenuar
    # baja frecuencia. Tolerancia: 0.7x random como margen seguro (en general bn ~ 0.10-0.20).
    assert lf_bn < 0.30, f"blue-noise lowfreq ratio inesperadamente alto: {lf_bn:.4f}"
    assert lf_bn < lf_random * 0.7, (
        f"blue-noise no es claramente superior a random (bn={lf_bn:.4f}, random={lf_random:.4f})"
    )


def test_blue_noise_beats_cluster_threshold() -> None:
    """
    Versus un threshold field 'cluster' (mitad arriba blanco), el blue-noise
    debe ser claramente mejor en baja frecuencia.
    """
    lbn = _load("laser_blue_noise")
    size = 16
    t_bn = lbn.threshold_matrix_for_dithering(size=size, seed=12345)
    gray = np.full((96, 96), 128, dtype=np.uint8) / 255.0
    tile = np.tile(t_bn, (96 // size + 1, 96 // size + 1))[:96, :96]
    out_bn = np.where(gray >= tile, 255, 0).astype(np.uint8)

    out_cluster = np.zeros((96, 96), dtype=np.uint8)
    out_cluster[:48, :] = 255

    lf_bn = _spectral_lowfreq_ratio(out_bn)
    lf_cl = _spectral_lowfreq_ratio(out_cluster)
    assert lf_bn < lf_cl, f"blue-noise debe ganar a cluster (bn={lf_bn:.4f}, cl={lf_cl:.4f})"
    assert lf_cl > 0.5, f"cluster grande deberia tener lf alto, got {lf_cl:.4f}"


def test_invalid_size_raises() -> None:
    lbn = _load("laser_blue_noise")
    with pytest.raises(ValueError):
        lbn.generate_void_and_cluster(size=0)
    with pytest.raises(ValueError):
        lbn.generate_void_and_cluster(size=1)
