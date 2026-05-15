"""
Generador de matrices blue-noise por void-and-cluster (Ulichney 1993).

Sustituye al `NOISE_16` ad-hoc de `laser_target_match.py`: blue-noise auténtico
distribuye energía uniformemente en alta frecuencia con ~0 energía en baja,
lo que elimina los "clusters direccionales" característicos de error-diffusion
en grabado laser horizontal.

Algoritmo (Ulichney 1993, "The Void-and-Cluster Method for Dither Array Generation"):

1. Initial Binary Pattern (IBP): empezar con patrón aleatorio con ratio bajo, luego
   estabilizar swappeando el "tightest cluster" (densest "1") por el "largest void"
   (least dense "0") hasta que no haya swaps utiles.
2. Phase 1: remover los "1"s de uno en uno, asignando rangos en orden descendente
   (el cluster más denso recibe el rango más alto entre los "1"s del IBP).
3. Phase 2 + 3: añadir "1"s a los voids más grandes restantes, asignando rangos
   ascendentes desde `n_ones` hasta `n - 1`.

La matriz resultante `R` con valores en `[0, size*size)` se convierte en threshold
field `T = (R + 0.5) / (size*size)` para ordered dithering.

Cache: la generación es O(N²) con Gaussian filter por iteración. Para `size=32`
toma ~5s; se persiste a `assets/blue_noise_{size}.npy` y se reusa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _REPO_ROOT / "assets"


def _blurred_density(binary: np.ndarray, sigma: float) -> np.ndarray:
    """Densidad local via Gaussian wrap (tile-able)."""
    return gaussian_filter(binary.astype(np.float64), sigma=sigma, mode="wrap")


def _argmax_where(arr: np.ndarray, mask: np.ndarray) -> int:
    """argmax(arr) restringido a posiciones con mask=True."""
    masked = np.where(mask, arr, -np.inf)
    return int(np.argmax(masked))


def _argmin_where(arr: np.ndarray, mask: np.ndarray) -> int:
    masked = np.where(mask, arr, np.inf)
    return int(np.argmin(masked))


def _stabilize_initial_binary_pattern(
    binary: np.ndarray, sigma: float, max_iters: int = 5000
) -> np.ndarray:
    """
    Estabiliza el IBP swappeando tightest cluster <-> largest void hasta no haber mejora.

    Returns: binario estabilizado (mismo shape, mismo numero de unos).
    """
    M = binary.copy()
    for _ in range(max_iters):
        ones_mask = (M == 1)
        density = _blurred_density(M, sigma)
        cluster_idx = _argmax_where(density, ones_mask)
        # remover cluster
        M_temp = M.copy()
        M_temp.flat[cluster_idx] = 0
        zeros_mask = (M_temp == 0)
        density_after = _blurred_density(M_temp, sigma)
        void_idx = _argmin_where(density_after, zeros_mask)
        if void_idx == cluster_idx:
            break  # estable: el void coincide con el cluster ya removido
        M.flat[cluster_idx] = 0
        M.flat[void_idx] = 1
    return M


def generate_void_and_cluster(
    size: int = 32,
    *,
    sigma: float = 1.5,
    initial_fill: float = 0.1,
    seed: int = 12345,
    verbose: bool = False,
) -> np.ndarray:
    """
    Genera matriz blue-noise por void-and-cluster (Ulichney 1993).

    Args:
        size: lado de la matriz (32 default; 64 es referencia ImageMagick pero ~10x mas lento).
        sigma: sigma del Gaussian para densidad local (Ulichney usa 1.5; aumentar suaviza patron).
        initial_fill: fraccion inicial de "1"s (0.1 trabaja bien para el IBP).
        seed: reproducible.
        verbose: imprime progreso por fases.

    Returns:
        np.ndarray shape (size, size) dtype int32 con ranks en [0, size*size).
        Cada valor unico. Para usar como threshold matrix: `(rank + 0.5) / (size*size)`.
    """
    if size <= 1:
        raise ValueError(f"size debe ser > 1, got {size}")
    rng = np.random.default_rng(seed)
    n = size * size
    n_ones = max(1, int(round(n * float(initial_fill))))

    # IBP
    M = np.zeros((size, size), dtype=np.int8)
    flat_idx = rng.permutation(n)[:n_ones]
    M.flat[flat_idx] = 1
    M = _stabilize_initial_binary_pattern(M, sigma=sigma)
    if verbose:
        print(f"[VAC] IBP estabilizado: n_ones={n_ones}/{n}")

    rank = np.full((size, size), -1, dtype=np.int32)

    # Phase 1: descontar "1"s del IBP en orden de tightest cluster -> rangos [0, n_ones)
    M1 = M.copy()
    for r in range(n_ones - 1, -1, -1):
        ones_mask = (M1 == 1)
        density = _blurred_density(M1, sigma)
        cluster_idx = _argmax_where(density, ones_mask)
        rank.flat[cluster_idx] = r
        M1.flat[cluster_idx] = 0
    if verbose:
        print(f"[VAC] Phase 1 listo: ranks 0..{n_ones-1} asignados")

    # Phase 2: agregar "1"s a voids restantes -> rangos [n_ones, n)
    M2 = M.copy()
    for r in range(n_ones, n):
        zeros_mask = (M2 == 0)
        density = _blurred_density(M2, sigma)
        void_idx = _argmin_where(density, zeros_mask)
        rank.flat[void_idx] = r
        M2.flat[void_idx] = 1
    if verbose:
        print(f"[VAC] Phase 2 listo: ranks {n_ones}..{n-1} asignados")

    if (rank < 0).any():
        raise RuntimeError("Bug: hay celdas sin asignar rank tras Phase 2")

    return rank


def void_and_cluster_matrix(
    size: int = 32,
    *,
    cache_dir: Path | None = None,
    force_regen: bool = False,
    **gen_kwargs,
) -> np.ndarray:
    """
    Matriz blue-noise cacheada en disco. Generada en primera invocacion.

    Args:
        size: lado.
        cache_dir: directorio para `.npy` (default `<repo>/assets`).
        force_regen: ignora cache y regenera.
        **gen_kwargs: pasados a `generate_void_and_cluster` (sigma, seed, etc.).
    """
    cache_dir = cache_dir or _CACHE_DIR
    cache_file = cache_dir / f"blue_noise_{size}.npy"
    if not force_regen and cache_file.is_file():
        arr = np.load(cache_file)
        if arr.shape == (size, size):
            return arr.astype(np.int32)
    matrix = generate_void_and_cluster(size=size, **gen_kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, matrix)
    return matrix


def threshold_matrix_for_dithering(
    size: int = 32, **kwargs
) -> np.ndarray:
    """
    Matriz threshold float [0,1] para ordered dithering.

    Uso en pipeline:
        T = threshold_matrix_for_dithering(32)
        out = np.where(gray >= T[i % 32, j % 32] * 255, 255, 0)
    """
    rank = void_and_cluster_matrix(size=size, **kwargs)
    return (rank.astype(np.float64) + 0.5) / float(size * size)
