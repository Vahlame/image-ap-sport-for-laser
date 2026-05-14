#!/usr/bin/env python3
"""
Busca parámetros/algoritmos que aproximen una imagen objetivo de referencia.

Entrada:
  - --input: foto original
  - --target: salida deseada (por ejemplo, referencia ImagR)

Salida:
  - match.sqlite: parámetros, métricas y archivo de cada intento
  - match_manifest.jsonl
  - match_XXXX.png
  - index.html con tarjetas ordenadas por score
  - contact_sheet.png con los mejores candidatos

Exploración guiada (opcional): --guided-explore con dedupe por hash, plateau (ventana de std),
reinicio perturbado y --batch-early-stop para lotes (p.ej. 20k) con checkpoints cada 5k.

Preprocesado (una ruta por corrida): none|sauvola|niblack|grabcut|watershed|chanvese|deeplab|unet|sam2.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from html import escape
import json
import sqlite3
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import uniform_filter
from scipy.stats import qmc

import laser_plateau
import laser_scoring

try:
    import numba
except ImportError:
    numba = None

try:
    import cv2
except ImportError:
    cv2 = None

# Máximo por una sola corrida: menos de 1 millón de candidatos (RAM/tiempo razonables).
MAX_CANDIDATES_PER_RUN = 999_999

# Exploración guiada (lotes / plateau): valores por defecto al usar --guided-explore.
DEFAULT_GUIDED_BATCH_SIZE = 20_000
DEFAULT_GUIDED_CHECKPOINT_EVERY = 5_000
DEFAULT_GUIDED_BATCH_EPSILON = 0.008
DEFAULT_PLATEAU_WINDOW = 500
DEFAULT_PLATEAU_STD_MAX = 0.008
DEFAULT_RESTART_CANDIDATES = 800
DEFAULT_EVAL_CHUNK = 128

RESTART_ALGORITHMS = (
    "floyd",
    "floyd_serpentine",
    "burkes",
    "burkes_serpentine",
    "sierra3",
    "sierra3_serpentine",
    "jarvis",
    "jarvis_serpentine",
    "stucki",
    "stucki_serpentine",
    "atkinson",
    "atkinson_serpentine",
    "floyd_midtones_bayer_shadows",
    "burkes_blue_mix",
    "two_pass_blue_then_sierra3",
    "blue_noise16",
    "bayer8",
)


BAYER_4 = (1 / 17.0) * np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float64,
)

BAYER_8 = (1 / 65.0) * np.array(
    [
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21],
    ],
    dtype=np.float64,
)


def ranked_noise_matrix(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.random((size, size))
    ranks = np.empty_like(values)
    ranks.flat[np.argsort(values, axis=None)] = np.arange(size * size, dtype=np.float64)
    return (ranks + 0.5) / float(size * size)


NOISE_16 = ranked_noise_matrix(16, 424242)

FLOYD_KERNEL = [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]
ATKINSON_KERNEL = [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]
JARVIS_KERNEL = [
    (1, 0, 7),
    (2, 0, 5),
    (-2, 1, 3),
    (-1, 1, 5),
    (0, 1, 7),
    (1, 1, 5),
    (2, 1, 3),
    (-2, 2, 1),
    (-1, 2, 3),
    (0, 2, 5),
    (1, 2, 3),
    (2, 2, 1),
]
STUCKI_KERNEL = [
    (1, 0, 8),
    (2, 0, 4),
    (-2, 1, 2),
    (-1, 1, 4),
    (0, 1, 8),
    (1, 1, 4),
    (2, 1, 2),
    (-2, 2, 1),
    (-1, 2, 2),
    (0, 2, 4),
    (1, 2, 2),
    (2, 2, 1),
]
BURKES_KERNEL = [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)]
SIERRA3_KERNEL = [(1, 0, 5), (2, 0, 3), (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2), (-1, 2, 2), (0, 2, 3), (1, 2, 2)]
SIERRA2_KERNEL = [(1, 0, 4), (2, 0, 3), (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)]
SIERRA_LITE_KERNEL = [(1, 0, 2), (-1, 1, 1), (0, 1, 1)]

DIFFUSION_ALGORITHMS: dict[str, tuple[list[tuple[int, int, float]], float, bool]] = {
    "floyd": (FLOYD_KERNEL, 16, False),
    "floyd_serpentine": (FLOYD_KERNEL, 16, True),
    "atkinson": (ATKINSON_KERNEL, 8, False),
    "atkinson_serpentine": (ATKINSON_KERNEL, 8, True),
    "jarvis": (JARVIS_KERNEL, 48, False),
    "jarvis_serpentine": (JARVIS_KERNEL, 48, True),
    "stucki": (STUCKI_KERNEL, 42, False),
    "stucki_serpentine": (STUCKI_KERNEL, 42, True),
    "burkes": (BURKES_KERNEL, 32, False),
    "burkes_serpentine": (BURKES_KERNEL, 32, True),
    "sierra3": (SIERRA3_KERNEL, 32, False),
    "sierra3_serpentine": (SIERRA3_KERNEL, 32, True),
    "sierra2": (SIERRA2_KERNEL, 16, False),
    "sierra2_serpentine": (SIERRA2_KERNEL, 16, True),
    "sierra_lite": (SIERRA_LITE_KERNEL, 4, False),
    "sierra_lite_serpentine": (SIERRA_LITE_KERNEL, 4, True),
}


_WORK_BASE_GRAY: np.ndarray | None = None
_WORK_TARGET_GRAY: np.ndarray | None = None
_WORK_TARGET_BINARY: np.ndarray | None = None
_WORK_TARGET_DENSITY: np.ndarray | None = None
_WORK_TARGET_EDGES: np.ndarray | None = None
_WORK_SCORE_VERSION: str = "v1"


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    invert: bool
    threshold: int
    contrast: float
    brightness: float
    gamma: float
    autocontrast: float
    sharpen: float


@dataclass
class MatchResult:
    id: int
    algorithm: str
    invert: bool
    threshold: int
    contrast: float
    brightness: float
    gamma: float
    autocontrast: float
    sharpen: float
    score: float
    pixel_error: float
    edge_error: float
    white_ratio: float
    target_white_ratio: float
    output_file: str
    seconds: float


def candidate_from_row(row: sqlite3.Row) -> Candidate:
    return Candidate(
        algorithm=str(row["algorithm"]),
        invert=bool(row["invert"]),
        threshold=int(row["threshold"]),
        contrast=float(row["contrast"]),
        brightness=float(row["brightness"]),
        gamma=float(row["gamma"]),
        autocontrast=float(row["autocontrast"]),
        sharpen=float(row["sharpen"]),
    )


def init_worker(
    base_gray: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    score_version: str = "v1",
) -> None:
    global _WORK_BASE_GRAY
    global _WORK_TARGET_GRAY
    global _WORK_TARGET_BINARY
    global _WORK_TARGET_DENSITY
    global _WORK_TARGET_EDGES
    global _WORK_SCORE_VERSION
    _WORK_BASE_GRAY = base_gray
    _WORK_TARGET_GRAY = target_gray
    _WORK_TARGET_BINARY = target_binary
    _WORK_TARGET_DENSITY = target_density
    _WORK_TARGET_EDGES = target_edges
    _WORK_SCORE_VERSION = str(score_version)


def default_worker_count() -> int:
    """Usa todos los núcleos lógicos disponibles (sin cap artificial)."""
    return max(1, os.cpu_count() or 2)


def parallel_chunksize(num_tasks: int, worker_count: int) -> int:
    if num_tasks <= 0:
        return 1
    return max(1, min(1024, num_tasks // max(1, worker_count * 4)))


def evaluate_candidate_task(task: tuple[int, Candidate]) -> tuple[int, Candidate, np.ndarray, float, float, float, float, float]:
    if (
        _WORK_BASE_GRAY is None
        or _WORK_TARGET_GRAY is None
        or _WORK_TARGET_BINARY is None
        or _WORK_TARGET_DENSITY is None
        or _WORK_TARGET_EDGES is None
    ):
        raise RuntimeError("Worker no inicializado")
    idx, candidate = task
    t0 = time.perf_counter()
    output = render_candidate(_WORK_BASE_GRAY, candidate)
    score, pixel_error, edge_error, white_ratio = laser_scoring.score_candidate_dispatch(
        _WORK_SCORE_VERSION,  # type: ignore[arg-type]
        output,
        _WORK_TARGET_GRAY,
        _WORK_TARGET_BINARY,
        _WORK_TARGET_DENSITY,
        _WORK_TARGET_EDGES,
        candidate,
    )
    elapsed = time.perf_counter() - t0
    return idx, candidate, output, score, pixel_error, edge_error, white_ratio, elapsed


def candidate_param_hash(candidate: Candidate) -> str:
    """Hash estable de parámetros (dedupe entre listas / reinicios)."""
    key = "|".join(
        (
            candidate.algorithm,
            str(int(candidate.invert)),
            str(candidate.threshold),
            f"{candidate.contrast:.6f}",
            f"{candidate.brightness:.6f}",
            f"{candidate.gamma:.6f}",
            f"{candidate.autocontrast:.6f}",
            f"{candidate.sharpen:.6f}",
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def dedupe_candidates_by_hash(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        h = candidate_param_hash(candidate)
        if h in seen:
            continue
        seen.add(h)
        out.append(candidate)
    skipped = len(candidates) - len(out)
    return out, skipped


def perturb_candidates_restart(
    base_gray: np.ndarray,
    target_white_ratio: float,
    count: int,
    rng: np.random.Generator,
    anchor: Candidate | None,
) -> list[Candidate]:
    """Genera candidatos aleatorios alrededor de un ancla para reinicio tras plateau."""
    otsu = int(otsu_threshold(base_gray))
    quantile_threshold = int(np.clip(np.quantile(base_gray, 1.0 - target_white_ratio), 1, 254))
    center = float(anchor.threshold) if anchor is not None else float((otsu + quantile_threshold) / 2.0)
    out: list[Candidate] = []
    attempts = 0
    max_attempts = max(count * 8, count + 50)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        thr = int(np.clip(rng.normal(center, 14.0), 1, 254))
        cand = Candidate(
            str(rng.choice(RESTART_ALGORITHMS)),
            bool(rng.integers(0, 2)),
            thr,
            float(rng.choice((0.52, 0.62, 0.72, 0.82, 0.92, 1.05, 1.2))),
            float(rng.normal(16.0, 10.0)),
            float(rng.choice((0.78, 0.88, 0.96, 1.0, 1.08, 1.18, 1.28))),
            float(rng.choice((1.0, 2.0, 3.0, 6.0))),
            float(rng.choice((0.0, 40.0, 90.0, 140.0))),
        )
        out.append(cand)
    return out


def score_window_std(window: deque[float]) -> float:
    if len(window) < 2:
        return float("inf")
    return float(np.std(np.array(window, dtype=np.float64), ddof=0))


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def otsu_threshold(gray: np.ndarray) -> int:
    clipped = np.clip(gray, 0, 255).astype(np.uint8)
    hist = np.bincount(clipped.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    bins = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * bins)
    sum_total = sum_bg[-1]
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.zeros_like(bins)
    mean_fg = np.zeros_like(bins)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between))


def detect_binary_target(target_gray: np.ndarray, tolerance: float = 0.95) -> bool:
    """True si la imagen objetivo ya es casi binaria (alto contraste global)."""
    near_black = int(np.sum(target_gray < 16))
    near_white = int(np.sum(target_gray > 239))
    ratio = (near_black + near_white) / max(1, int(target_gray.size))
    return ratio >= float(tolerance)


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def resize_to_target(input_image: Image.Image, target_size: tuple[int, int], max_side: int) -> Image.Image:
    if max_side > 0:
        scale = min(max_side / max(target_size), 1.0)
        target_size = (max(1, round(target_size[0] * scale)), max(1, round(target_size[1] * scale)))
    return ImageOps.fit(input_image, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def preprocess_gray(base_gray: np.ndarray, candidate: Candidate) -> np.ndarray:
    gray = base_gray.copy()
    if candidate.autocontrast > 0:
        low = np.percentile(gray, candidate.autocontrast)
        high = np.percentile(gray, 100.0 - candidate.autocontrast)
        if high > low:
            gray = (gray - low) * (255.0 / (high - low))
    gray = np.clip(gray, 0.0, 255.0)
    if candidate.gamma != 1.0:
        gray = 255.0 * np.power(gray / 255.0, 1.0 / candidate.gamma)
    gray = (gray - 128.0) * candidate.contrast + 128.0 + candidate.brightness
    gray = np.clip(gray, 0.0, 255.0)
    if candidate.sharpen > 0:
        image = Image.fromarray(gray.astype(np.uint8), mode="L")
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(candidate.sharpen), threshold=2))
        gray = np.array(image, dtype=np.float64)
    if candidate.invert:
        gray = 255.0 - gray
    return np.clip(gray, 0.0, 255.0)


def compose_masked_high_background(
    gray: np.ndarray,
    mask01: np.ndarray,
    *,
    feather: int = 5,
    background: float = 255.0,
    debug_path: Path | None = None,
) -> np.ndarray:
    """Combina luminancia con máscara 0..1: primer plano conservado, fondo → background (p.ej. blanco)."""
    g = gray.astype(np.float64)
    m = np.clip(np.asarray(mask01, dtype=np.float64), 0.0, 1.0)
    if feather > 0:
        k = max(1, int(feather))
        m = uniform_filter(m, size=k, mode="nearest")
        m = np.clip(m, 0.0, 1.0)
    out = g * m + (1.0 - m) * float(background)
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(m * 255.0, 0, 255).astype(np.uint8), mode="L").save(debug_path, optimize=True)
    return np.clip(out, 0.0, 255.0)


def sauvola_preprocess_gray(
    gray: np.ndarray,
    window: int,
    k: float,
    R: float,
    blend: float,
) -> np.ndarray:
    """Ajuste local tipo Sauvola antes de difusión: realza contraste respecto al umbral local T."""
    w = max(3, int(window) | 1)
    if w % 2 == 0:
        w += 1
    g = gray.astype(np.float64)
    m = uniform_filter(g, size=w, mode="nearest")
    m2 = uniform_filter(g * g, size=w, mode="nearest")
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    R = max(float(R), 1e-6)
    k = float(np.clip(k, 0.01, 0.99))
    T = m * (1.0 + k * (s / R - 1.0))
    enhanced = np.clip(g + 0.35 * (g - T), 0.0, 255.0)
    b = float(np.clip(blend, 0.0, 1.0))
    out = (1.0 - b) * g + b * enhanced
    return np.clip(out, 0.0, 255.0)


def niblack_preprocess_gray(gray: np.ndarray, window: int, k: float, blend: float) -> np.ndarray:
    """Umbral local Niblack: T = m + k*s (k suele ser negativo para texto oscuro sobre claro)."""
    w = max(3, int(window) | 1)
    if w % 2 == 0:
        w += 1
    g = gray.astype(np.float64)
    m = uniform_filter(g, size=w, mode="nearest")
    m2 = uniform_filter(g * g, size=w, mode="nearest")
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    k = float(np.clip(float(k), -0.5, 0.5))
    T = m + float(k) * s
    enhanced = np.clip(g + 0.35 * (g - T), 0.0, 255.0)
    b = float(np.clip(blend, 0.0, 1.0))
    out = (1.0 - b) * g + b * enhanced
    return np.clip(out, 0.0, 255.0)


def grabcut_preprocess_gray(
    rgb: np.ndarray,
    gray: np.ndarray,
    rect: tuple[int, int, int, int] | None,
    *,
    feather: int = 5,
    debug_path: Path | None = None,
) -> np.ndarray:
    """Máscara sujeto/fondo (GrabCut) y fondo claro para halftone centrado en la mano."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no instalado: pip install opencv-python")
    h, w = rgb.shape[:2]
    if rect is None:
        m = max(6, min(h, w) // 12)
        rect = (m, m, max(1, w - 2 * m), max(1, h - 2 * m))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mbin = np.where((mask_gc == cv2.GC_BGD) | (mask_gc == cv2.GC_PR_BGD), 0.0, 1.0).astype(np.float64)
    return compose_masked_high_background(gray, mbin, feather=feather, debug_path=debug_path)


def watershed_preprocess_gray(gray: np.ndarray) -> np.ndarray:
    """Watershed sobre gradiente con marcadores en máximos locales de DT (dedos tocándose)."""
    from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt, maximum_filter
    from skimage import filters, segmentation

    g = gray.astype(np.float64) / 255.0
    thr = float(filters.threshold_otsu(g))
    fg1 = g < thr
    fg2 = g >= thr
    fg = fg1
    for cand in (fg1, fg2):
        a = float(np.mean(cand))
        if 0.07 < a < 0.86:
            fg = cand
            break
    fg = binary_opening(fg, iterations=1)
    fg = binary_closing(fg, iterations=1)
    dt = distance_transform_edt(fg)
    min_dist = max(5, min(dt.shape) // 40)
    foot = np.ones((min_dist * 2 + 1, min_dist * 2 + 1), dtype=bool)
    local_max = (dt == maximum_filter(dt, footprint=foot, mode="nearest")) & (dt >= 0.8) & fg
    coords = np.argwhere(local_max)
    markers = np.zeros(g.shape, dtype=np.int32)
    markers[~fg] = 1
    if coords.size > 0:
        for label_id, rc in enumerate(coords, start=2):
            markers[int(rc[0]), int(rc[1])] = label_id
    else:
        rr, cc = np.nonzero(fg)
        if rr.size > 0:
            cy, cx = int(np.mean(rr)), int(np.mean(cc))
            markers[cy, cx] = 2
    grad = filters.sobel(g)
    labels = segmentation.watershed(grad, markers, mask=fg)
    mod = labels.astype(np.float64)
    mod = mod - float(np.mean(mod[fg])) if np.any(fg) else mod * 0.0
    out = np.clip(gray.astype(np.float64) + mod * 0.55, 0.0, 255.0)
    return out


def chan_vese_preprocess_gray(
    gray: np.ndarray,
    num_iter: int,
    smoothing: int,
    log_iters: bool,
    *,
    feather: int = 5,
    debug_path: Path | None = None,
) -> np.ndarray:
    """Chan–Vese morfológico (skimage): región vs fondo; máscara suavizada y fondo claro."""
    from skimage.segmentation import morphological_chan_vese

    g = gray.astype(np.float64) / 255.0
    t0 = time.perf_counter()
    progress = {"it": 0}

    def _cb(_ls: np.ndarray) -> None:
        progress["it"] += 1
        it = progress["it"]
        step = max(2, max(1, num_iter // 6))
        if log_iters and it > 0 and (it == num_iter or it % step == 0):
            elapsed = time.perf_counter() - t0
            print(f"[CHANVESE] paso={it}/{num_iter} t={elapsed:.2f}s", flush=True)

    seg = morphological_chan_vese(
        g,
        num_iter=int(num_iter),
        init_level_set="checkerboard",
        smoothing=int(smoothing),
        iter_callback=_cb,
    )
    mbin = np.clip(seg.astype(np.float64), 0.0, 1.0)
    out = compose_masked_high_background(gray, mbin, feather=feather, debug_path=debug_path)
    if log_iters:
        print(f"[CHANVESE] listo num_iter={num_iter} t_total={time.perf_counter() - t0:.2f}s", flush=True)
    return out


def deeplab_preprocess_gray(
    rgb: np.ndarray,
    gray: np.ndarray,
    device_str: str,
    min_side: int,
    class_index: int | None,
    *,
    feather: int = 5,
    debug_path: Path | None = None,
) -> np.ndarray:
    """DeepLabV3+ ResNet50 (torchvision, COCO→VOC): probabilidad de clase (p.ej. persona) como máscara suave."""
    try:
        import torch
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
    except ImportError as exc:
        raise RuntimeError("DeepLab requiere torch y torchvision: pip install torch torchvision") from exc

    if cv2 is None:
        raise RuntimeError("DeepLab usa OpenCV para resize: pip install opencv-python")

    dev = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[DEEPLAB] CUDA no disponible; usando CPU.", flush=True)

    h0, w0 = rgb.shape[:2]
    min_side_eff = max(128, min(int(min_side), max(h0, w0)))
    scale = max(float(min_side_eff), 1.0) / float(min(h0, w0))
    nh, nw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t0 = time.perf_counter()
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(dev).eval()
    cats = weights.meta.get("categories", [])
    if class_index is None:
        try:
            ci = cats.index("person") if "person" in cats else 15
        except ValueError:
            ci = 15
    else:
        ci = int(class_index)
    x = torch.from_numpy(rgb_s).permute(2, 0, 1).float().to(dev) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1)
    x = (x - mean) / std
    x = x.unsqueeze(0)
    with torch.no_grad():
        logits = model(x)["out"]
        nch = int(logits.shape[1])
        if ci < 0 or ci >= nch:
            raise RuntimeError(f"deeplab-class {ci} fuera de rango 0..{nch - 1} (salida tiene {nch} canales)")
        pr = torch.softmax(logits, dim=1)[0, ci].float().cpu().numpy()
    pr = cv2.resize(pr, (w0, h0), interpolation=cv2.INTER_LINEAR)
    out = compose_masked_high_background(gray, pr.astype(np.float64), feather=feather, debug_path=debug_path)
    nm = cats[ci] if 0 <= ci < len(cats) else "?"
    print(
        f"[DEEPLAB] clase_idx={ci} nombre={nm} min_side={min_side_eff} "
        f"t_infer={time.perf_counter() - t0:.2f}s device={dev}",
        flush=True,
    )
    return out


def _build_mini_unet() -> object:
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, inc: int, outc: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(inc, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True),
                nn.Conv2d(outc, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class MiniUNet(nn.Module):
        """U-Net pequeño 1→1 canal (debe coincidir con --unet-weights entrenado para esta arquitectura)."""

        def __init__(self) -> None:
            super().__init__()
            self.e1 = DoubleConv(1, 32)
            self.p1 = nn.MaxPool2d(2)
            self.e2 = DoubleConv(32, 64)
            self.p2 = nn.MaxPool2d(2)
            self.e3 = DoubleConv(64, 128)
            self.p3 = nn.MaxPool2d(2)
            self.b = DoubleConv(128, 256)
            self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.d3 = DoubleConv(256, 128)
            self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.d2 = DoubleConv(128, 64)
            self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d1 = DoubleConv(64, 32)
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            c1 = self.e1(x)
            c2 = self.e2(self.p1(c1))
            c3 = self.e3(self.p2(c2))
            b = self.b(self.p3(c3))
            x3 = self.u3(b)
            x3 = self.d3(torch.cat([x3, c3], dim=1))
            x2 = self.u2(x3)
            x2 = self.d2(torch.cat([x2, c2], dim=1))
            x1 = self.u1(x2)
            x1 = self.d1(torch.cat([x1, c1], dim=1))
            return self.out(x1)

    return MiniUNet()


def unet_preprocess_gray(
    gray: np.ndarray,
    weights_path: Path,
    device_str: str,
    threshold: float,
    *,
    feather: int = 5,
    debug_path: Path | None = None,
) -> np.ndarray:
    """Inferencia MiniUNet 1 canal; pesos con la misma arquitectura que _build_mini_unet()."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("U-Net requiere torch: pip install torch") from exc

    if not weights_path.is_file():
        raise RuntimeError(f"No existe --unet-weights: {weights_path}")

    dev = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[UNET] CUDA no disponible; usando CPU.", flush=True)

    model = _build_mini_unet().to(dev).eval()
    try:
        state = torch.load(str(weights_path), map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(str(weights_path), map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[UNET] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    h0, w0 = gray.shape
    mult = 8
    ph = (mult - h0 % mult) % mult
    pw = (mult - w0 % mult) % mult
    g = np.pad(gray.astype(np.float64), ((0, ph), (0, pw)), mode="reflect") / 255.0
    t0 = time.perf_counter()
    x = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    prob = prob[:h0, :w0]
    mbin = np.clip(prob.astype(np.float64), 0.0, 1.0)
    if threshold > 0.0:
        mbin = (mbin >= float(threshold)).astype(np.float64)
    out = compose_masked_high_background(gray, mbin, feather=feather, debug_path=debug_path)
    print(f"[UNET] t_infer={time.perf_counter() - t0:.2f}s device={dev} weights={weights_path.name}", flush=True)
    return out


def sam2_preprocess_gray(
    rgb: np.ndarray,
    gray: np.ndarray,
    prompts_path: Path,
    model_id: str,
    device_str: str,
    multimask_output: bool,
    mask_index: int,
    feather: int,
    debug_path: Path | None,
) -> np.ndarray:
    """SAM 2 (Hugging Face Transformers): segmentación por caja/puntos en JSON; máscara → fondo claro."""
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ImportError as exc:
        raise RuntimeError(
            'SAM 2 requiere: pip install "transformers>=4.46.0" torch torchvision accelerate'
        ) from exc

    if not prompts_path.is_file():
        raise RuntimeError(f"No existe --sam2-prompts: {prompts_path}")

    raw = json.loads(prompts_path.read_text(encoding="utf-8"))
    has_box = bool(raw.get("input_boxes"))
    has_pts = bool(raw.get("input_points"))
    if not has_box and not has_pts:
        raise ValueError("SAM2 JSON: define input_boxes (caja) y/o input_points con input_labels.")
    if has_pts and "input_labels" not in raw:
        raise ValueError("SAM2 JSON: input_labels obligatorio cuando hay input_points.")

    mm = bool(raw.get("multimask_output", multimask_output))
    midx = int(raw.get("mask_index", mask_index))

    image_pil = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id)
    dev = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[SAM2] CUDA no disponible; usando CPU.", flush=True)
    model = model.to(dev).eval()

    proc_kwargs: dict = {"images": image_pil, "return_tensors": "pt"}
    if has_box:
        proc_kwargs["input_boxes"] = raw["input_boxes"]
    if has_pts:
        proc_kwargs["input_points"] = raw["input_points"]
        proc_kwargs["input_labels"] = raw["input_labels"]

    t0 = time.perf_counter()
    inputs = processor(**proc_kwargs)
    inputs = inputs.to(dev)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=mm)
    t_inf = time.perf_counter() - t0

    pred = outputs.pred_masks.cpu()
    orig_sizes = inputs["original_sizes"]
    reshaped = inputs["reshaped_input_sizes"] if "reshaped_input_sizes" in inputs else None
    try:
        if reshaped is not None:
            masks_list = processor.post_process_masks(pred, orig_sizes, reshaped)
        else:
            masks_list = processor.post_process_masks(pred, orig_sizes)
    except TypeError:
        masks_list = processor.post_process_masks(pred, orig_sizes)

    m0 = masks_list[0]
    if isinstance(m0, torch.Tensor):
        arr = m0.detach().float().cpu().numpy()
    else:
        arr = np.asarray(m0, dtype=np.float64)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        k = arr.shape[0]
        sel = arr[int(midx) % k]
    elif arr.ndim == 2:
        sel = arr
    else:
        raise RuntimeError(f"SAM2: forma de máscara inesperada {arr.shape}")

    sel = sel.astype(np.float64)
    if sel.size > 0 and (sel.max() > 1.0 + 1e-3 or sel.min() < -1e-3):
        sel = 1.0 / (1.0 + np.exp(-np.clip(sel, -30.0, 30.0)))
    sel = np.clip(sel, 0.0, 1.0)

    h0, w0 = gray.shape
    if sel.shape != (h0, w0):
        if cv2 is None:
            raise RuntimeError("SAM2: OpenCV necesario para alinear máscara al tamaño de gray.")
        sel = cv2.resize(sel.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)

    print(
        "[SAM2] Licencia: revisar términos Meta SAM2 / Hugging Face para uso comercial. "
        f"modelo={model_id} multimask={mm} t_infer={t_inf:.2f}s device={dev}",
        flush=True,
    )
    return compose_masked_high_background(gray, sel, feather=feather, debug_path=debug_path)


def parse_grabcut_rect(text: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    t = text.strip()
    if not t:
        return None
    parts = [int(x) for x in t.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("grabcut-rect debe ser x,y,w,h con cuatro enteros")
    x, y, rw, rh = parts
    if rw < 2 or rh < 2 or x < 0 or y < 0 or x + rw > width or y + rh > height:
        raise ValueError("grabcut-rect fuera de imagen o tamaño invalido")
    return (x, y, rw, rh)


def ordered_dither(gray: np.ndarray, threshold: int, matrix: np.ndarray, strength: float = 64.0) -> np.ndarray:
    h, w = gray.shape
    tile = np.tile(matrix, (h // matrix.shape[0] + 1, w // matrix.shape[1] + 1))[:h, :w]
    local_threshold = threshold + (tile - 0.5) * strength
    return np.where(gray >= local_threshold, 255, 0).astype(np.uint8)


def midtone_mask(gray: np.ndarray, low: float = 52.0, high: float = 204.0) -> np.ndarray:
    return (gray >= low) & (gray <= high)


# Variantes discretas de Burkes+blue-noise (misma firma Candidate; distinta mezcla espacial / fuerza ordered).
BURKES_BLUE_VARIANTS: dict[str, tuple[float, float, float]] = {
    "burkes_blue_mix": (48.0, 210.0, 72.0),
    "burkes_blue_mix_narrow": (58.0, 198.0, 72.0),
    "burkes_blue_mix_wide": (35.0, 228.0, 72.0),
    "burkes_blue_mix_softblue": (48.0, 210.0, 58.0),
    "burkes_blue_mix_hardblue": (48.0, 210.0, 90.0),
    "burkes_blue_mix_tightmid": (62.0, 188.0, 72.0),
}

# Familia acrylic / blue-noise + difusión y multi-pasada (lista larga; la rejilla densa reparte en round-robin).
BLUE_DENSE_ALGOS: tuple[str, ...] = (
    "burkes_blue_mix",
    "burkes_blue_mix_narrow",
    "burkes_blue_mix_wide",
    "burkes_blue_mix_softblue",
    "burkes_blue_mix_hardblue",
    "burkes_blue_mix_tightmid",
    "sierra3_blue_mix",
    "two_pass_blue_then_sierra3",
    "burkes_serpentine",
    "sierra3_serpentine",
    "sierra2_serpentine",
    "sierra3",
    "sierra2",
    "floyd_serpentine",
    "atkinson_serpentine",
    "jarvis_serpentine",
    "stucki_serpentine",
    "blue_noise16",
    "sierra3_midtones_blue_extremes",
    "floyd_bayer8_mix",
    "jarvis_bayer8_edge_mix",
    "jarvis_midtones_threshold_extremes",
    "floyd_midtones_bayer_shadows",
    "atkinson_highlights_stucki_shadows",
    "two_pass_bayer_then_floyd",
    "two_pass_threshold_then_jarvis",
    "two_pass_soft_edges_atkinson",
)


def dense_blue_family_candidates(
    quantile_threshold: int,
    otsu: int,
    cap: int,
    sampling: str = "sobol",
    seed: int = 42,
) -> list[Candidate]:
    """Rejilla o muestra Sobol (6D + round-robin de algoritmo); `cap` nunca supera MAX_CANDIDATES_PER_RUN."""
    cap = min(MAX_CANDIDATES_PER_RUN, max(1, cap))
    anchors = {int(np.clip(x, 1, 254)) for x in (quantile_threshold, otsu, 87, 82, 94, 76, 102, 110, 118)}
    thr_set: set[int] = set()
    for center in anchors:
        for delta in range(-28, 29, 2):
            thr_set.add(int(np.clip(center + delta, 1, 254)))
    thresholds = sorted(thr_set)

    scale = max(40.0, float(cap) ** 0.34)
    n_thr = max(16, min(220, int(scale * 1.1)))
    if len(thresholds) > n_thr:
        step = max(1, len(thresholds) // n_thr)
        thresholds = thresholds[::step][:n_thr]

    n_con = max(5, min(26, 5 + int(cap / 55_000)))
    n_br = max(5, min(24, 5 + int(cap / 60_000)))
    n_g = max(5, min(22, 5 + int(cap / 65_000)))
    contrasts = [round(0.36 + (0.42 * i / max(1, n_con - 1)), 3) for i in range(n_con)]
    brightnesses = [round(8.0 + (30.0 * i / max(1, n_br - 1)), 3) for i in range(n_br)]
    gammas = [round(0.82 + (0.46 * i / max(1, n_g - 1)), 3) for i in range(n_g)]
    autocontrasts = tuple(round(x, 3) for x in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0))[: max(3, min(7, 3 + cap // 200_000))]
    sharpens = (0.0, 40.0, 80.0, 120.0, 170.0)[: max(2, min(5, 2 + cap // 150_000))]
    inverts = (True, False) if cap >= 50_000 else (True,)

    algorithms = BLUE_DENSE_ALGOS
    out: list[Candidate] = []
    rr = 0

    if sampling == "grid":
        for threshold in thresholds:
            for contrast in contrasts:
                for brightness in brightnesses:
                    for gamma in gammas:
                        for autocontrast in autocontrasts:
                            for sharpen in sharpens:
                                for invert in inverts:
                                    algorithm = algorithms[rr % len(algorithms)]
                                    rr += 1
                                    out.append(
                                        Candidate(
                                            algorithm,
                                            invert,
                                            threshold,
                                            contrast,
                                            brightness,
                                            gamma,
                                            float(autocontrast),
                                            float(sharpen),
                                        )
                                    )
                                    if len(out) >= cap:
                                        return out
        return out

    rng = np.random.default_rng(seed)
    sobol_gen = qmc.Sobol(d=6, scramble=True, seed=int(rng.integers(1, 2**30)))
    n_pts = min(cap, max(256, cap))
    # Sobol balanceado en scipy: n debe ser potencia de 2; recortamos a n_pts.
    n_draw = 1 << ((int(n_pts) - 1).bit_length()) if int(n_pts) > 1 else 1
    pts = sobol_gen.random(n=n_draw)[:n_pts]
    n_t, n_c, n_b, n_g, n_a, n_s = (
        len(thresholds),
        len(contrasts),
        len(brightnesses),
        len(gammas),
        len(autocontrasts),
        len(sharpens),
    )
    for row in pts:
        ti = min(n_t - 1, int(np.floor(float(row[0]) * n_t)))
        ci = min(n_c - 1, int(np.floor(float(row[1]) * n_c)))
        bi = min(n_br - 1, int(np.floor(float(row[2]) * n_br)))
        gi = min(n_g - 1, int(np.floor(float(row[3]) * n_g)))
        ai = min(n_a - 1, int(np.floor(float(row[4]) * n_a)))
        si = min(n_s - 1, int(np.floor(float(row[5]) * n_s)))
        invert = inverts[rr % len(inverts)]
        algorithm = algorithms[rr % len(algorithms)]
        rr += 1
        out.append(
            Candidate(
                algorithm,
                invert,
                int(thresholds[ti]),
                float(contrasts[ci]),
                float(brightnesses[bi]),
                float(gammas[gi]),
                float(autocontrasts[ai]),
                float(sharpens[si]),
            )
        )
        if len(out) >= cap:
            return out
    return out


def shadow_mask(gray: np.ndarray, threshold: float = 92.0) -> np.ndarray:
    return gray < threshold


def highlight_mask(gray: np.ndarray, threshold: float = 172.0) -> np.ndarray:
    return gray > threshold


_ERROR_DIFFUSION_JIT = None

if numba is not None:
    try:

        @numba.njit(cache=True)
        def _error_diffusion_jit_core(
            work: np.ndarray,
            out: np.ndarray,
            threshold: int,
            dx: np.ndarray,
            dy: np.ndarray,
            wt: np.ndarray,
            divisor: float,
            serpentine: bool,
        ) -> None:
            h, w = work.shape
            nk = dx.shape[0]
            for y in range(h):
                if serpentine and (y & 1):
                    x = w - 1
                    direction = -1
                    while x >= 0:
                        old = work[y, x]
                        new = 255.0 if old >= threshold else 0.0
                        out[y, x] = 255 if new > 0.0 else 0
                        err = old - new
                        for k in range(nk):
                            nx = x + dx[k] * direction
                            ny = y + dy[k]
                            if 0 <= nx < w and 0 <= ny < h:
                                work[ny, nx] += err * wt[k] / divisor
                        x -= 1
                else:
                    x = 0
                    direction = 1
                    while x < w:
                        old = work[y, x]
                        new = 255.0 if old >= threshold else 0.0
                        out[y, x] = 255 if new > 0.0 else 0
                        err = old - new
                        for k in range(nk):
                            nx = x + dx[k] * direction
                            ny = y + dy[k]
                            if 0 <= nx < w and 0 <= ny < h:
                                work[ny, nx] += err * wt[k] / divisor
                        x += 1

        setattr(sys.modules[__name__], "_ERROR_DIFFUSION_JIT", _error_diffusion_jit_core)
    except Exception as exc:
        print(f"[NUMBA] error_diffusion JIT no activado: {exc}", flush=True)
        setattr(sys.modules[__name__], "_ERROR_DIFFUSION_JIT", None)


def _error_diffusion_dispatch_jit(
    gray: np.ndarray,
    threshold: int,
    kernel: list[tuple[int, int, float]],
    divisor: float,
    serpentine: bool,
) -> np.ndarray:
    work = gray.astype(np.float64).copy()
    h, w = work.shape
    out = np.zeros((h, w), dtype=np.uint8)
    dx = np.array([k[0] for k in kernel], dtype=np.int32)
    dy = np.array([k[1] for k in kernel], dtype=np.int32)
    wt = np.array([k[2] for k in kernel], dtype=np.float64)
    assert _ERROR_DIFFUSION_JIT is not None
    _ERROR_DIFFUSION_JIT(work, out, int(threshold), dx, dy, wt, float(divisor), bool(serpentine))
    return out


def error_diffusion(
    gray: np.ndarray,
    threshold: int,
    kernel: list[tuple[int, int, float]],
    divisor: float,
    serpentine: bool = False,
) -> np.ndarray:
    if _ERROR_DIFFUSION_JIT is not None:
        return _error_diffusion_dispatch_jit(gray, threshold, kernel, divisor, serpentine)
    return _error_diffusion_python(gray, threshold, kernel, divisor, serpentine)


def _error_diffusion_python(
    gray: np.ndarray,
    threshold: int,
    kernel: list[tuple[int, int, float]],
    divisor: float,
    serpentine: bool = False,
) -> np.ndarray:
    work = gray.astype(np.float64).copy()
    h, w = work.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        xs = range(w - 1, -1, -1) if serpentine and y % 2 else range(w)
        direction = -1 if serpentine and y % 2 else 1
        for x in xs:
            old = work[y, x]
            new = 255.0 if old >= threshold else 0.0
            out[y, x] = 255 if new > 0 else 0
            err = old - new
            for dx, dy, weight in kernel:
                nx = x + dx * direction
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    work[ny, nx] += err * weight / divisor
    return out


def render_candidate(base_gray: np.ndarray, candidate: Candidate) -> np.ndarray:
    gray = preprocess_gray(base_gray, candidate)
    if candidate.algorithm == "threshold":
        return np.where(gray >= candidate.threshold, 255, 0).astype(np.uint8)
    if candidate.algorithm == "bayer4":
        return ordered_dither(gray, candidate.threshold, BAYER_4, strength=72.0)
    if candidate.algorithm == "bayer8":
        return ordered_dither(gray, candidate.threshold, BAYER_8, strength=72.0)
    if candidate.algorithm == "blue_noise16":
        return ordered_dither(gray, candidate.threshold, NOISE_16, strength=72.0)
    if candidate.algorithm in DIFFUSION_ALGORITHMS:
        kernel, divisor, serpentine = DIFFUSION_ALGORITHMS[candidate.algorithm]
        return error_diffusion(gray, candidate.threshold, kernel, divisor, serpentine=serpentine)
    if candidate.algorithm == "sierra3_blue_mix":
        sierra = error_diffusion(gray, candidate.threshold, SIERRA3_KERNEL, 32, serpentine=True)
        blue = ordered_dither(gray, candidate.threshold, NOISE_16, strength=78.0)
        return np.where(edge_map(gray) > 0.12, sierra, blue).astype(np.uint8)
    if candidate.algorithm in BURKES_BLUE_VARIANTS:
        mid_lo, mid_hi, blue_strength = BURKES_BLUE_VARIANTS[candidate.algorithm]
        burkes = error_diffusion(gray, candidate.threshold, BURKES_KERNEL, 32, serpentine=True)
        blue = ordered_dither(gray, candidate.threshold, NOISE_16, strength=blue_strength)
        return np.where((gray >= mid_lo) & (gray <= mid_hi), burkes, blue).astype(np.uint8)
    if candidate.algorithm == "sierra3_midtones_blue_extremes":
        sierra = error_diffusion(gray, candidate.threshold, SIERRA3_KERNEL, 32, serpentine=True)
        blue = ordered_dither(gray, candidate.threshold, NOISE_16, strength=92.0)
        hard = np.where(gray >= candidate.threshold, 255, 0).astype(np.uint8)
        extremes = shadow_mask(gray, 58.0) | highlight_mask(gray, 214.0)
        return np.where(extremes, hard, np.where(midtone_mask(gray), sierra, blue)).astype(np.uint8)
    if candidate.algorithm == "two_pass_blue_then_sierra3":
        textured = ordered_dither(gray, candidate.threshold, NOISE_16, strength=54.0).astype(np.float64)
        blended = np.clip(gray * 0.74 + textured * 0.26, 0, 255)
        return error_diffusion(blended, candidate.threshold, SIERRA3_KERNEL, 32, serpentine=True)
    if candidate.algorithm == "floyd_bayer8_mix":
        floyd = error_diffusion(gray, candidate.threshold, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16)
        bayer = ordered_dither(gray, candidate.threshold, BAYER_8, strength=72.0)
        mask = edge_map(gray) > 0.18
        return np.where(mask, floyd, bayer).astype(np.uint8)
    if candidate.algorithm == "stucki_bayer8_mix":
        stucki = error_diffusion(
            gray,
            candidate.threshold,
            [
                (1, 0, 8),
                (2, 0, 4),
                (-2, 1, 2),
                (-1, 1, 4),
                (0, 1, 8),
                (1, 1, 4),
                (2, 1, 2),
                (-2, 2, 1),
                (-1, 2, 2),
                (0, 2, 4),
                (1, 2, 2),
                (2, 2, 1),
            ],
            42,
        )
        bayer = ordered_dither(gray, candidate.threshold, BAYER_8, strength=72.0)
        mask = edge_map(gray) > 0.18
        return np.where(mask, stucki, bayer).astype(np.uint8)
    if candidate.algorithm == "jarvis_bayer8_edge_mix":
        jarvis = error_diffusion(
            gray,
            candidate.threshold,
            [
                (1, 0, 7),
                (2, 0, 5),
                (-2, 1, 3),
                (-1, 1, 5),
                (0, 1, 7),
                (1, 1, 5),
                (2, 1, 3),
                (-2, 2, 1),
                (-1, 2, 3),
                (0, 2, 5),
                (1, 2, 3),
                (2, 2, 1),
            ],
            48,
        )
        bayer = ordered_dither(gray, candidate.threshold, BAYER_8, strength=72.0)
        return np.where(edge_map(gray) > 0.15, jarvis, bayer).astype(np.uint8)
    if candidate.algorithm == "atkinson_bayer8_edge_mix":
        atkinson = error_diffusion(gray, candidate.threshold, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8)
        bayer = ordered_dither(gray, candidate.threshold, BAYER_8, strength=72.0)
        return np.where(edge_map(gray) > 0.15, atkinson, bayer).astype(np.uint8)
    if candidate.algorithm == "jarvis_midtones_threshold_extremes":
        jarvis = error_diffusion(
            gray,
            candidate.threshold,
            [
                (1, 0, 7),
                (2, 0, 5),
                (-2, 1, 3),
                (-1, 1, 5),
                (0, 1, 7),
                (1, 1, 5),
                (2, 1, 3),
                (-2, 2, 1),
                (-1, 2, 3),
                (0, 2, 5),
                (1, 2, 3),
                (2, 2, 1),
            ],
            48,
        )
        hard = np.where(gray >= candidate.threshold, 255, 0).astype(np.uint8)
        return np.where(midtone_mask(gray), jarvis, hard).astype(np.uint8)
    if candidate.algorithm == "floyd_midtones_bayer_shadows":
        floyd = error_diffusion(gray, candidate.threshold, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16)
        bayer = ordered_dither(gray, candidate.threshold, BAYER_8, strength=84.0)
        return np.where(midtone_mask(gray), floyd, bayer).astype(np.uint8)
    if candidate.algorithm == "atkinson_highlights_stucki_shadows":
        atkinson = error_diffusion(gray, candidate.threshold, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8)
        stucki = error_diffusion(
            gray,
            candidate.threshold,
            [
                (1, 0, 8),
                (2, 0, 4),
                (-2, 1, 2),
                (-1, 1, 4),
                (0, 1, 8),
                (1, 1, 4),
                (2, 1, 2),
                (-2, 2, 1),
                (-1, 2, 2),
                (0, 2, 4),
                (1, 2, 2),
                (2, 2, 1),
            ],
            42,
        )
        return np.where(highlight_mask(gray), atkinson, stucki).astype(np.uint8)
    if candidate.algorithm == "two_pass_bayer_then_floyd":
        textured = ordered_dither(gray, candidate.threshold, BAYER_8, strength=48.0).astype(np.float64)
        blended = np.clip(gray * 0.72 + textured * 0.28, 0, 255)
        return error_diffusion(blended, candidate.threshold, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16)
    if candidate.algorithm == "two_pass_threshold_then_jarvis":
        hard = np.where(gray >= candidate.threshold, 255.0, 0.0)
        blended = np.clip(gray * 0.65 + hard * 0.35, 0, 255)
        return error_diffusion(
            blended,
            candidate.threshold,
            [
                (1, 0, 7),
                (2, 0, 5),
                (-2, 1, 3),
                (-1, 1, 5),
                (0, 1, 7),
                (1, 1, 5),
                (2, 1, 3),
                (-2, 2, 1),
                (-1, 2, 3),
                (0, 2, 5),
                (1, 2, 3),
                (2, 2, 1),
            ],
            48,
        )
    if candidate.algorithm == "two_pass_soft_edges_atkinson":
        edges = edge_map(gray)
        softened = np.clip(gray + (edges - 0.5) * 42.0, 0, 255)
        return error_diffusion(softened, candidate.threshold, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8)
    if candidate.algorithm == "floyd":
        return error_diffusion(gray, candidate.threshold, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16)
    if candidate.algorithm == "atkinson":
        return error_diffusion(gray, candidate.threshold, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8)
    if candidate.algorithm == "jarvis":
        return error_diffusion(
            gray,
            candidate.threshold,
            [
                (1, 0, 7),
                (2, 0, 5),
                (-2, 1, 3),
                (-1, 1, 5),
                (0, 1, 7),
                (1, 1, 5),
                (2, 1, 3),
                (-2, 2, 1),
                (-1, 2, 3),
                (0, 2, 5),
                (1, 2, 3),
                (2, 2, 1),
            ],
            48,
        )
    if candidate.algorithm == "stucki":
        return error_diffusion(
            gray,
            candidate.threshold,
            [
                (1, 0, 8),
                (2, 0, 4),
                (-2, 1, 2),
                (-1, 1, 4),
                (0, 1, 8),
                (1, 1, 4),
                (2, 1, 2),
                (-2, 2, 1),
                (-1, 2, 2),
                (0, 2, 4),
                (1, 2, 2),
                (2, 2, 1),
            ],
            42,
        )
    raise ValueError(f"Algoritmo no soportado: {candidate.algorithm}")


def edge_map(gray: np.ndarray) -> np.ndarray:
    y_grad, x_grad = np.gradient(gray.astype(np.float64) / 255.0)
    mag = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    p95 = np.percentile(mag, 95)
    if p95 > 0:
        mag = mag / p95
    return np.clip(mag, 0.0, 1.0)


def density_map(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = gray.shape
    small_size = (max(1, w // scale), max(1, h // scale))
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    return np.array(image.resize(small_size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0


def score_candidate(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compatibilidad: delega en scoring v1 (misma semantica historica)."""
    return laser_scoring.score_candidate_v1(out, target_gray, target_binary, target_density, target_edges)


def build_candidates(
    base_gray: np.ndarray,
    target_white_ratio: float,
    limit: int,
    sampling: str = "sobol",
    sobol_seed: int = 42,
) -> list[Candidate]:
    otsu = otsu_threshold(base_gray)
    quantile_threshold = int(np.clip(np.quantile(base_gray, 1.0 - target_white_ratio), 1, 254))
    thresholds = sorted(
        {
            int(np.clip(t, 1, 254))
            for center in (otsu, quantile_threshold, 96, 112, 128, 144, 160)
            for t in (center - 18, center - 9, center, center + 9, center + 18)
        }
    )
    algorithms = [
        "threshold",
        "bayer4",
        "bayer8",
        "blue_noise16",
        "floyd",
        "floyd_serpentine",
        "atkinson",
        "atkinson_serpentine",
        "jarvis",
        "jarvis_serpentine",
        "stucki",
        "stucki_serpentine",
        "burkes",
        "burkes_serpentine",
        "sierra3",
        "sierra3_serpentine",
        "sierra2",
        "sierra2_serpentine",
        "sierra_lite",
        "sierra_lite_serpentine",
        "sierra3_blue_mix",
        "burkes_blue_mix",
        "burkes_blue_mix_narrow",
        "burkes_blue_mix_wide",
        "burkes_blue_mix_softblue",
        "burkes_blue_mix_hardblue",
        "burkes_blue_mix_tightmid",
        "sierra3_midtones_blue_extremes",
        "two_pass_blue_then_sierra3",
        "floyd_bayer8_mix",
        "stucki_bayer8_mix",
        "jarvis_bayer8_edge_mix",
        "atkinson_bayer8_edge_mix",
        "jarvis_midtones_threshold_extremes",
        "floyd_midtones_bayer_shadows",
        "atkinson_highlights_stucki_shadows",
        "two_pass_bayer_then_floyd",
        "two_pass_threshold_then_jarvis",
        "two_pass_soft_edges_atkinson",
    ]
    base_candidates: list[Candidate] = []
    for threshold in thresholds:
        for algorithm in algorithms:
            for invert in (False, True):
                for contrast in (0.85, 1.15, 1.55):
                    for brightness in (-14.0, 0.0, 14.0):
                        base_candidates.append(Candidate(algorithm, invert, threshold, contrast, brightness, 1.0, 1.0, 0.0))

    # Segunda capa: opciones más agresivas, útiles cuando la referencia parece tener textura tipo acrylic.
    aggressive_candidates: list[Candidate] = []
    for threshold in (quantile_threshold - 18, quantile_threshold - 9, quantile_threshold, quantile_threshold + 9, quantile_threshold + 18, otsu):
        for algorithm in (
            "floyd",
            "floyd_serpentine",
            "atkinson",
            "atkinson_serpentine",
            "jarvis",
            "jarvis_serpentine",
            "stucki",
            "stucki_serpentine",
            "burkes",
            "burkes_serpentine",
            "sierra3",
            "sierra3_serpentine",
            "sierra2",
            "sierra2_serpentine",
            "sierra_lite",
            "sierra_lite_serpentine",
            "blue_noise16",
            "sierra3_blue_mix",
            "burkes_blue_mix",
            "burkes_blue_mix_narrow",
            "burkes_blue_mix_wide",
            "burkes_blue_mix_softblue",
            "burkes_blue_mix_hardblue",
            "burkes_blue_mix_tightmid",
            "sierra3_midtones_blue_extremes",
            "two_pass_blue_then_sierra3",
            "bayer8",
            "floyd_bayer8_mix",
            "stucki_bayer8_mix",
            "jarvis_bayer8_edge_mix",
            "atkinson_bayer8_edge_mix",
            "jarvis_midtones_threshold_extremes",
            "floyd_midtones_bayer_shadows",
            "atkinson_highlights_stucki_shadows",
            "two_pass_bayer_then_floyd",
            "two_pass_threshold_then_jarvis",
            "two_pass_soft_edges_atkinson",
        ):
            for invert in (True, False):
                for contrast in (0.85, 1.15, 1.45):
                    for brightness in (-18.0, 0.0, 18.0):
                        for gamma in (0.7, 0.85, 1.15, 1.45):
                            for autocontrast in (1.0, 3.0, 6.0):
                                for sharpen in (0.0, 90.0, 180.0):
                                    aggressive_candidates.append(
                                        Candidate(
                                            algorithm,
                                            invert,
                                            int(np.clip(threshold, 1, 254)),
                                            contrast,
                                            brightness,
                                            gamma,
                                            autocontrast,
                                            sharpen,
                                        )
                                    )

    # Tercera capa: búsqueda local alrededor de los rangos que suelen parecerse al target ImagR/Acrylic.
    focused_candidates: list[Candidate] = []
    focused_algorithms = (
        "floyd",
        "floyd_serpentine",
        "stucki",
        "stucki_serpentine",
        "jarvis",
        "jarvis_serpentine",
        "atkinson",
        "atkinson_serpentine",
        "burkes",
        "burkes_serpentine",
        "sierra3",
        "sierra3_serpentine",
        "sierra2",
        "sierra2_serpentine",
        "sierra_lite",
        "sierra_lite_serpentine",
        "blue_noise16",
        "sierra3_blue_mix",
        "burkes_blue_mix",
        "burkes_blue_mix_narrow",
        "burkes_blue_mix_wide",
        "burkes_blue_mix_softblue",
        "burkes_blue_mix_hardblue",
        "burkes_blue_mix_tightmid",
        "sierra3_midtones_blue_extremes",
        "two_pass_blue_then_sierra3",
        "floyd_bayer8_mix",
        "stucki_bayer8_mix",
        "jarvis_bayer8_edge_mix",
        "atkinson_bayer8_edge_mix",
        "jarvis_midtones_threshold_extremes",
        "floyd_midtones_bayer_shadows",
        "atkinson_highlights_stucki_shadows",
        "two_pass_bayer_then_floyd",
        "two_pass_threshold_then_jarvis",
        "two_pass_soft_edges_atkinson",
    )
    for threshold in range(max(1, quantile_threshold - 30), min(255, quantile_threshold + 31), 5):
        for contrast in (0.55, 0.62, 0.68, 0.75, 0.82, 0.9):
            for brightness in (10.0, 16.0, 20.0, 24.0, 30.0):
                for gamma in (0.82, 0.92, 1.0, 1.08, 1.15):
                    for algorithm in focused_algorithms:
                        focused_candidates.append(
                            Candidate(
                                algorithm,
                                True,
                                threshold,
                                contrast,
                                brightness,
                                gamma,
                                2.0,
                                0.0,
                            )
                        )

    # Capa densa: factor extra para deduplicación al mezclar buckets; nunca supera MAX_CANDIDATES_PER_RUN.
    dense_cap = min(MAX_CANDIDATES_PER_RUN, max(1, max(limit, int(limit * 1.35))))
    dense_candidates = dense_blue_family_candidates(quantile_threshold, otsu, dense_cap, sampling, sobol_seed)

    buckets = [dense_candidates, focused_candidates, aggressive_candidates, base_candidates]
    candidates: list[Candidate] = []
    max_len = max(len(bucket) for bucket in buckets)
    for i in range(max_len):
        for bucket in buckets:
            if i < len(bucket):
                candidates.append(bucket[i])

    unique: list[Candidate] = []
    seen: set[Candidate] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def focused_candidates(center_threshold: int, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    focused_algorithms = (
        "floyd",
        "floyd_serpentine",
        "stucki",
        "stucki_serpentine",
        "jarvis",
        "jarvis_serpentine",
        "atkinson",
        "atkinson_serpentine",
        "burkes",
        "burkes_serpentine",
        "sierra3",
        "sierra3_serpentine",
        "sierra2",
        "sierra2_serpentine",
        "sierra_lite",
        "sierra_lite_serpentine",
        "blue_noise16",
        "sierra3_blue_mix",
        "burkes_blue_mix",
        "burkes_blue_mix_narrow",
        "burkes_blue_mix_wide",
        "burkes_blue_mix_softblue",
        "burkes_blue_mix_hardblue",
        "burkes_blue_mix_tightmid",
        "sierra3_midtones_blue_extremes",
        "two_pass_blue_then_sierra3",
        "bayer8",
        "floyd_bayer8_mix",
        "stucki_bayer8_mix",
        "jarvis_bayer8_edge_mix",
        "atkinson_bayer8_edge_mix",
        "jarvis_midtones_threshold_extremes",
        "floyd_midtones_bayer_shadows",
        "atkinson_highlights_stucki_shadows",
        "two_pass_bayer_then_floyd",
        "two_pass_threshold_then_jarvis",
        "two_pass_soft_edges_atkinson",
    )
    for threshold in range(max(1, center_threshold - 22), min(255, center_threshold + 23), 3):
        for contrast in (0.52, 0.58, 0.65, 0.72, 0.78, 0.85):
            for brightness in (12.0, 18.0, 22.0, 26.0, 30.0):
                for gamma in (0.85, 0.94, 1.0, 1.06, 1.12):
                    for algorithm in focused_algorithms:
                        candidates.append(Candidate(algorithm, True, threshold, contrast, brightness, gamma, 2.0, 0.0))
    unique: list[Candidate] = []
    seen: set[Candidate] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def prepare_output_dir(out_dir: Path, resume: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    for pattern in ("match_*.png", "match.sqlite", "match_manifest.jsonl", "index.html", "contact_sheet.png", "target_binary.png"):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    thumbs = out_dir / "thumbs"
    if thumbs.is_dir():
        for path in thumbs.glob("*.png"):
            if path.is_file():
                path.unlink()
        try:
            thumbs.rmdir()
        except OSError:
            pass


def ensure_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            invert INTEGER NOT NULL,
            threshold INTEGER NOT NULL,
            contrast REAL NOT NULL,
            brightness REAL NOT NULL,
            gamma REAL NOT NULL,
            autocontrast REAL NOT NULL,
            sharpen REAL NOT NULL,
            score REAL NOT NULL,
            pixel_error REAL NOT NULL,
            edge_error REAL NOT NULL,
            white_ratio REAL NOT NULL,
            target_white_ratio REAL NOT NULL,
            output_file TEXT NOT NULL,
            seconds REAL NOT NULL,
            created TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_score ON matches(score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_algorithm ON matches(algorithm)")
    conn.commit()
    return conn


def read_top_candidates(db_path: Path, top_k: int, best_per_algorithm: bool = False) -> list[Candidate]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if best_per_algorithm:
            rows = conn.execute(
                """
                SELECT algorithm, invert, threshold, contrast, brightness, gamma, autocontrast, sharpen
                FROM (
                    SELECT
                        algorithm, invert, threshold, contrast, brightness, gamma, autocontrast, sharpen, score,
                        ROW_NUMBER() OVER (PARTITION BY algorithm ORDER BY score ASC) AS rank_in_algorithm
                    FROM matches
                )
                WHERE rank_in_algorithm <= ?
                ORDER BY score ASC
                """,
                (top_k,),
            ).fetchall()
            return [candidate_from_row(row) for row in rows]
        rows = conn.execute(
            """
            SELECT algorithm, invert, threshold, contrast, brightness, gamma, autocontrast, sharpen
            FROM matches
            ORDER BY score ASC
            LIMIT ?
            """,
            (top_k,),
        ).fetchall()
    return [candidate_from_row(row) for row in rows]


def neighbor_algorithms(algorithm: str) -> tuple[str, ...]:
    sierra_family = (
        "sierra3_serpentine",
        "two_pass_blue_then_sierra3",
        "sierra3_blue_mix",
        "sierra3_midtones_blue_extremes",
        "sierra3",
        "sierra2_serpentine",
        "sierra2",
        "burkes_serpentine",
        "burkes",
        "burkes_blue_mix",
        "two_pass_bayer_then_floyd",
        "burkes_blue_mix_narrow",
        "burkes_blue_mix_wide",
        "burkes_blue_mix_softblue",
        "burkes_blue_mix_hardblue",
        "burkes_blue_mix_tightmid",
    )
    if "sierra" in algorithm or "burkes" in algorithm or "blue" in algorithm:
        return sierra_family
    if "floyd" in algorithm:
        return (
            "floyd",
            "floyd_serpentine",
            "floyd_midtones_bayer_shadows",
            "two_pass_bayer_then_floyd",
            "sierra3_serpentine",
            "sierra3_blue_mix",
        )
    if "jarvis" in algorithm:
        return (
            "jarvis",
            "jarvis_serpentine",
            "jarvis_midtones_threshold_extremes",
            "two_pass_threshold_then_jarvis",
            "sierra3_serpentine",
        )
    return (algorithm, "sierra3_serpentine", "burkes_serpentine", "two_pass_blue_then_sierra3")


def ordered_unique(values: list[float] | list[int]) -> list:
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def local_refine_candidates(db_path: Path, top_k: int, limit: int, best_per_algorithm: bool = False) -> list[Candidate]:
    anchors = read_top_candidates(db_path, top_k, best_per_algorithm)
    candidates: list[Candidate] = []
    for anchor in anchors:
        thresholds = ordered_unique(
            [
                int(np.clip(anchor.threshold + delta, 1, 254))
                for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -8, 8)
            ]
        )
        contrasts = ordered_unique(
            [round(max(0.35, anchor.contrast + delta), 3) for delta in (0.0, -0.02, 0.02, -0.04, 0.04, -0.08, 0.08)]
        )
        brightnesses = ordered_unique(
            [round(anchor.brightness + delta, 3) for delta in (0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -3.0, 3.0)]
        )
        gammas = ordered_unique(
            [round(max(0.45, anchor.gamma + delta), 3) for delta in (0.0, -0.04, 0.04, -0.06, 0.06, -0.12, 0.12)]
        )
        autocontrasts = ordered_unique([round(max(0.0, anchor.autocontrast + delta), 3) for delta in (0.0, -1.0, 1.0, 2.0)])
        sharpens = ordered_unique([anchor.sharpen, 0.0, 60.0, 100.0])
        algorithms = neighbor_algorithms(anchor.algorithm)
        for contrast in contrasts:
            for brightness in brightnesses:
                for gamma in gammas:
                    for autocontrast in autocontrasts:
                        for sharpen in sharpens:
                            for algorithm in algorithms:
                                for threshold in thresholds:
                                    candidates.append(
                                        Candidate(
                                            algorithm,
                                            anchor.invert,
                                            threshold,
                                            contrast,
                                            brightness,
                                            gamma,
                                            autocontrast,
                                            sharpen,
                                        )
                                    )
    unique: list[Candidate] = []
    seen: set[Candidate] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def make_thumbnail(src: Path, dst: Path, side: int) -> None:
    with Image.open(src) as im:
        thumb = im.convert("RGB")
        thumb.thumbnail((side, side), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (side, side), (15, 23, 42))
        canvas.paste(thumb, ((side - thumb.width) // 2, (side - thumb.height) // 2))
        canvas.save(dst, optimize=True)


def write_contact_sheet(out_dir: Path, rows: list[MatchResult], side: int, top_k: int) -> None:
    selected = rows[:top_k]
    cols = 5
    label_h = 72
    pad = 14
    cell_w = side + pad * 2
    cell_h = side + label_h + pad * 2
    rows_count = max(1, (len(selected) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_w, rows_count * cell_h), (8, 13, 23))
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(selected):
        x0 = (idx % cols) * cell_w + pad
        y0 = (idx // cols) * cell_h + pad
        with Image.open(out_dir / row.output_file) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((side, side), Image.Resampling.NEAREST)
            sheet.paste(thumb, (x0 + (side - thumb.width) // 2, y0 + (side - thumb.height) // 2))
        draw.text(
            (x0, y0 + side + 8),
            f"#{row.id:04d} {row.algorithm}\nscore {row.score:.4f} thr {row.threshold}\ninv {int(row.invert)} w {row.white_ratio:.2f}",
            fill=(226, 232, 240),
        )
    sheet.save(out_dir / "contact_sheet.png", optimize=True)


def write_html(out_dir: Path, rows: list[MatchResult], input_name: str, target_name: str, side: int) -> None:
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    for row in rows:
        make_thumbnail(out_dir / row.output_file, thumbs / row.output_file, side)

    algorithms = sorted({row.algorithm for row in rows})
    buttons = ['<button type="button" class="filter active" data-algorithm="all">Todos</button>']
    buttons.extend(
        f'<button type="button" class="filter" data-algorithm="{escape(algorithm)}">{escape(algorithm)}</button>'
        for algorithm in algorithms
    )
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card" data-algorithm="{escape(row.algorithm)}">
              <a href="{escape(row.output_file)}" target="_blank"><img src="thumbs/{escape(row.output_file)}" alt="match {row.id}" loading="lazy"></a>
              <div class="meta">
                <strong>#{row.id:04d}</strong><span class="pill">{escape(row.algorithm)}</span>
                <span>score <b>{row.score:.4f}</b></span><span>px {row.pixel_error:.4f}</span>
                <span>edge {row.edge_error:.4f}</span><span>white {row.white_ratio:.3f}</span>
                <span>thr {row.threshold}</span><span>inv {int(row.invert)}</span>
                <span>c {row.contrast:.2f}</span><span>b {row.brightness:+.0f}</span>
                <span>g {row.gamma:.2f}</span><span>sharp {row.sharpen:.0f}</span>
              </div>
            </article>
            """
        )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Target match láser</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070b12; --panel:#0f172a; --muted:#8b9cb3; --text:#e5edf7; --accent:#38bdf8; --border:rgba(148,163,184,.22); }}
    body {{ margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ position:sticky; top:0; z-index:5; padding:18px 22px; background:rgba(7,11,18,.9); backdrop-filter:blur(14px); border-bottom:1px solid var(--border); }}
    h1 {{ margin:0 0 8px; font-size:1.25rem; }}
    .summary,.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    .summary span {{ color:var(--muted); font-size:.9rem; }}
    button,a {{ border:1px solid var(--border); background:#111c30; color:var(--text); border-radius:999px; padding:8px 12px; cursor:pointer; text-decoration:none; font-weight:700; }}
    button.active,a.primary {{ border-color:var(--accent); color:#dff7ff; }}
    main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; padding:18px; }}
    .card {{ background:rgba(15,23,42,.86); border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
    .card img {{ width:100%; display:block; image-rendering:pixelated; background:#020617; }}
    .meta {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 8px; padding:10px; font-size:.82rem; color:var(--muted); }}
    .meta strong,.meta b {{ color:var(--text); }}
    .pill {{ color:var(--accent); font-weight:700; overflow-wrap:anywhere; }}
    .hidden {{ display:none; }}
  </style>
</head>
<body>
  <header>
    <h1>Target match láser — mejores primero</h1>
    <div class="summary">
      <span>Input: <b>{escape(input_name)}</b></span>
      <span>Target: <b>{escape(target_name)}</b></span>
      <span>{len(rows)} pruebas</span>
      <span>DB: match.sqlite</span>
      <span>Target binario: target_binary.png</span>
    </div>
    <div class="controls">
      {''.join(buttons)}
      <a class="primary" href="contact_sheet.png" target="_blank">Hoja top</a>
      <a href="target_binary.png" target="_blank">Ver target binario</a>
    </div>
  </header>
  <main>{''.join(cards)}</main>
  <script>
    const buttons = [...document.querySelectorAll('.filter')];
    const cards = [...document.querySelectorAll('.card')];
    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        for (const b of buttons) b.classList.remove('active');
        button.classList.add('active');
        const algorithm = button.dataset.algorithm;
        for (const card of cards) card.classList.toggle('hidden', algorithm !== 'all' && card.dataset.algorithm !== algorithm);
      }});
    }}
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def write_best_report(
    out_dir: Path,
    rows: list[MatchResult],
    input_name: str,
    target_name: str,
    meta_extra: dict | None = None,
) -> None:
    best = rows[0]
    payload = {
        "input": input_name,
        "target": target_name,
        "best_file": best.output_file,
        "best_path": str(out_dir / best.output_file),
        "score": best.score,
        "pixel_error": best.pixel_error,
        "edge_error": best.edge_error,
        "white_ratio": best.white_ratio,
        "target_white_ratio": best.target_white_ratio,
        "algorithm": best.algorithm,
        "invert": best.invert,
        "threshold": best.threshold,
        "contrast": best.contrast,
        "brightness": best.brightness,
        "gamma": best.gamma,
        "autocontrast": best.autocontrast,
        "sharpen": best.sharpen,
        "top": [asdict(row) for row in rows[:10]],
    }
    if meta_extra:
        payload.update(meta_extra)
    (out_dir / "best_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Best Match Report",
        "",
        f"- Input: `{input_name}`",
        f"- Target: `{target_name}`",
        f"- Best image: `{best.output_file}`",
        f"- Score: `{best.score:.6f}`",
        f"- Pixel error: `{best.pixel_error:.6f}`",
        f"- Edge error: `{best.edge_error:.6f}`",
        f"- White ratio: `{best.white_ratio:.6f}` (target `{best.target_white_ratio:.6f}`)",
        "",
        "## Winning Parameters",
        "",
        f"- Algorithm: `{best.algorithm}`",
        f"- Invert: `{int(best.invert)}`",
        f"- Threshold: `{best.threshold}`",
        f"- Contrast: `{best.contrast:.4f}`",
        f"- Brightness: `{best.brightness:+.4f}`",
        f"- Gamma: `{best.gamma:.4f}`",
        f"- Autocontrast: `{best.autocontrast:.4f}`",
        f"- Sharpen: `{best.sharpen:.4f}`",
        "",
        "## Top 10",
        "",
    ]
    for index, row in enumerate(rows[:10], start=1):
        lines.append(
            f"{index}. `{row.output_file}` score `{row.score:.6f}` | `{row.algorithm}` "
            f"thr `{row.threshold}` c `{row.contrast:.3f}` b `{row.brightness:+.1f}` g `{row.gamma:.3f}`"
        )
    (out_dir / "best_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def adaptive_epsilon(best_current: float, floor: float = 0.0005) -> float:
    """2% del mejor score global actual con piso (para checkpoints de lotes guiados)."""
    if not math.isfinite(best_current) or best_current == float("inf"):
        return floor
    return max(floor, float(best_current) * 0.02)


def guided_run_evaluation(
    args: argparse.Namespace,
    candidates: list[Candidate],
    base_gray: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    target_white_ratio: float,
    manifest,
    db: sqlite3.Connection,
    out_dir: Path,
) -> list[MatchResult]:
    """Evaluación por chunks: dedupe hash, plateau con reinicio perturbado, lotes 20k y early-stop cada 5k."""
    if args.dedupe_param_hashes:
        work_list, skipped_dup = dedupe_candidates_by_hash(candidates)
        if skipped_dup:
            print(f"Dedupe parametros (hash): omitidos {skipped_dup} duplicados.", flush=True)
    else:
        work_list = list(candidates)
    work: deque[Candidate] = deque(work_list)
    seen_eval_hashes: set[str] = set()

    results: list[MatchResult] = []
    eval_count = 0
    max_evals = args.n
    best_global = float("inf")
    best_anchor: Candidate | None = None
    rng = np.random.default_rng(args.explore_seed)

    plateau_on = args.plateau_detect
    print(
        f"[CONFIG] plateau_detect={plateau_on} window={args.plateau_window} std_max={args.plateau_std_max}",
        flush=True,
    )
    plateau_detector: laser_plateau.PlateauDetector | None = (
        laser_plateau.PlateauDetector(args.plateau_window, args.plateau_std_max) if plateau_on else None
    )
    batch_on = args.batch_early_stop
    batch_size = args.guided_batch_size
    checkpoint_every = args.guided_checkpoint_every
    batch_eps = args.guided_batch_epsilon
    restart_n = args.restart_candidates

    worker_count = args.workers if args.workers > 0 else default_worker_count()
    chunk_tasks_max = max(32, min(args.eval_chunk, 256))
    best_per_batch: list[float] = []
    batch_index = 0
    last_completed_batch_best: float | None = None

    sv = str(getattr(args, "score_version", "v1"))
    init_worker(base_gray, target_gray, target_binary, target_density, target_edges, sv)
    executor: ProcessPoolExecutor | None = None
    if worker_count > 1:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
            initargs=(base_gray, target_gray, target_binary, target_density, target_edges, sv),
        )

    def drain_chunk(task_pairs: list[tuple[int, Candidate]]) -> list[tuple[int, Candidate, np.ndarray, float, float, float, float, float]]:
        if not task_pairs:
            return []
        if worker_count == 1:
            return [evaluate_candidate_task(t) for t in task_pairs]
        assert executor is not None
        sub_chunk = max(1, parallel_chunksize(len(task_pairs), worker_count))
        return list(executor.map(evaluate_candidate_task, task_pairs, chunksize=sub_chunk))

    try:
        while work and eval_count < max_evals:
            if batch_on:
                batch_index += 1
                best_previous_batch = last_completed_batch_best
                snapshot_at_batch_start = best_global
                batch_this_best = float("inf")
                evals_in_current_batch = 0
                batch_queue: deque[Candidate] = deque()
                while work and len(batch_queue) < batch_size:
                    batch_queue.append(work.popleft())
                active_batch = batch_queue
            else:
                best_previous_batch = None
                snapshot_at_batch_start = float("nan")
                batch_this_best = float("inf")
                evals_in_current_batch = 0
                active_batch = work

            batch_aborted = False
            while active_batch and eval_count < max_evals:
                take = min(chunk_tasks_max, len(active_batch) if batch_on else len(work))
                if take <= 0:
                    break
                chunk_pairs: list[tuple[int, Candidate]] = []
                for _ in range(take):
                    cand = active_batch.popleft() if batch_on else work.popleft()
                    chunk_pairs.append((0, cand))

                for packed in drain_chunk(chunk_pairs):
                    _task_id, candidate, output, score, pixel_error, edge_error, white_ratio, elapsed = packed
                    seen_eval_hashes.add(candidate_param_hash(candidate))

                    if score < best_global:
                        best_global = score
                        best_anchor = candidate
                    batch_this_best = min(batch_this_best, score)
                    eval_count += 1
                    evals_in_current_batch += 1

                    filename = f"match_{eval_count:04d}.png"
                    Image.fromarray(output, mode="L").save(out_dir / filename, optimize=True)
                    row = MatchResult(
                        id=eval_count,
                        **asdict(candidate),
                        score=score,
                        pixel_error=pixel_error,
                        edge_error=edge_error,
                        white_ratio=white_ratio,
                        target_white_ratio=target_white_ratio,
                        output_file=filename,
                        seconds=elapsed,
                    )
                    results.append(row)
                    manifest.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    db.execute(
                        """
                        INSERT INTO matches (
                            algorithm, invert, threshold, contrast, brightness, gamma, autocontrast, sharpen,
                            score, pixel_error, edge_error, white_ratio, target_white_ratio, output_file, seconds, created
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        (
                            row.algorithm,
                            int(row.invert),
                            row.threshold,
                            row.contrast,
                            row.brightness,
                            row.gamma,
                            row.autocontrast,
                            row.sharpen,
                            row.score,
                            row.pixel_error,
                            row.edge_error,
                            row.white_ratio,
                            row.target_white_ratio,
                            row.output_file,
                            row.seconds,
                        ),
                    )
                    if eval_count % 50 == 0:
                        db.commit()
                        print(f"  {eval_count}/{max_evals} best={best_global:.4f}", flush=True)

                    if plateau_detector is not None:
                        act = plateau_detector.observe(score)
                        if act == laser_plateau.PlateauAction.RESTART:
                            print(
                                f"[PLATEAU] iter={eval_count} score_best={best_global:.4f} "
                                f"-> reiniciando exploracion (perturb)",
                                flush=True,
                            )
                            inject = perturb_candidates_restart(
                                base_gray,
                                target_white_ratio,
                                restart_n,
                                rng,
                                best_anchor,
                            )
                            added = 0
                            for c in inject:
                                ch = candidate_param_hash(c)
                                if ch in seen_eval_hashes:
                                    continue
                                if batch_on:
                                    active_batch.appendleft(c)
                                else:
                                    work.appendleft(c)
                                added += 1
                            print(f"  [PLATEAU] candidatos perturbados encolados: {added}", flush=True)

                    if batch_on and checkpoint_every > 0 and evals_in_current_batch > 0 and evals_in_current_batch % checkpoint_every == 0:
                        if math.isfinite(snapshot_at_batch_start):
                            improvement = snapshot_at_batch_start - batch_this_best
                        else:
                            improvement = float("inf")
                        delta_vs_global = batch_this_best - best_global
                        eps_use = (
                            adaptive_epsilon(best_global)
                            if str(getattr(args, "epsilon_mode", "adaptive")) == "adaptive"
                            else batch_eps
                        )
                        if improvement < eps_use:
                            decision = "ABORT_AND_RESTART"
                            reason = (
                                f"mejora_lote_vs_inicio={improvement:.4f} < epsilon={eps_use:.4f}"
                                if math.isfinite(snapshot_at_batch_start)
                                else "snapshot_no_finito"
                            )
                        else:
                            decision = "CONTINUE"
                            reason = f"mejora_lote_vs_inicio={improvement:.4f} >= epsilon={eps_use:.4f}"
                        print(
                            f"[CHECKPOINT] batch={batch_index} iter={evals_in_current_batch}/{batch_size}\n"
                            f"  best_this_batch={batch_this_best:.4f}\n"
                            f"  best_previous_batch={best_previous_batch}\n"
                            f"  best_global={best_global:.4f}\n"
                            f"  best_at_batch_start={snapshot_at_batch_start}\n"
                            f"  delta_vs_global_now={delta_vs_global:+.4f}\n"
                            f"  epsilon_effective={eps_use:.6f}\n"
                            f"  decision={decision} ({reason})",
                            flush=True,
                        )
                        if decision == "ABORT_AND_RESTART":
                            batch_aborted = True
                            while active_batch:
                                active_batch.popleft()
                            break

            if batch_on:
                last_batch_best = batch_this_best if math.isfinite(batch_this_best) else None
                if last_batch_best is not None:
                    best_per_batch.append(last_batch_best)
                    last_completed_batch_best = last_batch_best
                if batch_aborted:
                    print(
                        f"[BATCH] lote {batch_index} abortado tras checkpoint; "
                        f"historial mejores/lote (ultimos 8): {best_per_batch[-8:]}",
                        flush=True,
                    )
                else:
                    print(
                        f"[BATCH] lote {batch_index} completado; mejor_lote={last_batch_best}; "
                        f"historial (ultimos 8): {best_per_batch[-8:]}",
                        flush=True,
                    )

            if not batch_on:
                break

    finally:
        if executor is not None:
            executor.shutdown()

    db.commit()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Ajuste por búsqueda contra imagen target")
    parser.add_argument("--input", required=True, type=Path, help="Foto original")
    parser.add_argument("--target", required=True, type=Path, help="Imagen objetivo/reference")
    parser.add_argument("--out", required=True, type=Path, help="Carpeta de salida")
    parser.add_argument(
        "--n",
        type=int,
        default=600,
        help=f"Cantidad máxima de candidatos únicos por corrida (máx. {MAX_CANDIDATES_PER_RUN:,}; la rejilla densa escala con --n).",
    )
    parser.add_argument("--max-side", type=int, default=520, help="Lado máximo de comparación para acelerar")
    parser.add_argument("--top-report", type=int, default=80, help="Cuántos candidatos incluir en la galería")
    parser.add_argument("--thumb-side", type=int, default=180, help="Tamaño de miniatura")
    parser.add_argument("--focus-threshold", type=int, default=0, help="Si >0, busca solo alrededor de este umbral")
    parser.add_argument("--from-db", type=Path, default=None, help="SQLite previa: re-renderiza sus mejores candidatos")
    parser.add_argument("--from-db-top", type=int, default=40, help="Cantidad de candidatos a re-renderizar desde --from-db")
    parser.add_argument("--from-db-best-per-algorithm", action="store_true", help="Toma top N por cada algoritmo desde --from-db")
    parser.add_argument("--refine-db", type=Path, default=None, help="SQLite previa: expande vecindarios locales alrededor de sus mejores candidatos")
    parser.add_argument("--refine-top", type=int, default=3, help="Cantidad de anclas a expandir desde --refine-db")
    parser.add_argument("--refine-best-per-algorithm", action="store_true", help="Usa top N por algoritmo como anclas de --refine-db")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Procesos paralelos por imagen. 0 = todos los núcleos lógicos (os.cpu_count); 1 = secuencial/debug",
    )
    parser.add_argument("--resume", action="store_true", help="No limpiar salida previa")
    parser.add_argument(
        "--guided-explore",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Chunking: dedupe por hash, plateau con reinicio perturbado; con --batch-early-stop: lotes 20k y early-stop cada 5k.",
    )
    parser.add_argument(
        "--dedupe-param-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Omitir duplicados exactos de parametros antes de evaluar (solo modo guiado).",
    )
    parser.add_argument(
        "--plateau-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Si la std de los ultimos scores (ventana) es baja, encolar perturbacion aleatoria.",
    )
    parser.add_argument("--plateau-window", type=int, default=DEFAULT_PLATEAU_WINDOW, help="Ventana movil de scores para plateau")
    parser.add_argument("--plateau-std-max", type=float, default=DEFAULT_PLATEAU_STD_MAX, help="Umbral std (plateau si std < este valor)")
    parser.add_argument(
        "--batch-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lotes de --guided-batch-size; checkpoint cada --guided-checkpoint-every comparando mejora vs inicio de lote.",
    )
    parser.add_argument("--guided-batch-size", type=int, default=DEFAULT_GUIDED_BATCH_SIZE)
    parser.add_argument("--guided-checkpoint-every", type=int, default=DEFAULT_GUIDED_CHECKPOINT_EVERY)
    parser.add_argument("--guided-batch-epsilon", type=float, default=DEFAULT_GUIDED_BATCH_EPSILON)
    parser.add_argument("--explore-seed", type=int, default=42, help="Semilla para perturbaciones tras plateau")
    parser.add_argument("--restart-candidates", type=int, default=DEFAULT_RESTART_CANDIDATES)
    parser.add_argument(
        "--eval-chunk",
        type=int,
        default=DEFAULT_EVAL_CHUNK,
        help="Tamano de wave al ProcessPool en modo guiado",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("none", "sauvola", "niblack", "grabcut", "watershed", "chanvese", "deeplab", "unet", "sam2"),
        default="none",
        help="Pre-CV: sauvola|niblack|grabcut|watershed|chanvese|deeplab|unet|sam2 (HF; ver deps).",
    )
    parser.add_argument(
        "--preprocess-mask-feather",
        type=int,
        default=5,
        help="Radio suavizado uniforme de máscara (GrabCut/ChanVese/DeepLab/U-Net/SAM2).",
    )
    parser.add_argument(
        "--save-preprocess-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Guarda preprocess_mask.png (máscara suavizada) junto a preprocessed_gray.png.",
    )
    parser.add_argument("--sauvola-window", type=int, default=15, help="Ventana impar (Sauvola/Niblack)")
    parser.add_argument("--sauvola-k", type=float, default=0.15, help="k Sauvola (positivo) o Niblack (típ. -0.2 a 0.2)")
    parser.add_argument("--sauvola-r", type=float, default=128.0, help="R dinámico Sauvola (solo sauvola)")
    parser.add_argument("--sauvola-blend", type=float, default=0.35, help="0..1 mezcla con imagen original (sauvola/niblack)")
    parser.add_argument(
        "--grabcut-rect",
        type=str,
        default="",
        help="Rectángulo inicial x,y,w,h en píxeles (OpenCV); vacío = márgenes automáticos.",
    )
    parser.add_argument("--chanvese-iter", type=int, default=24, help="Iteraciones morphological_chan_vese")
    parser.add_argument("--chanvese-smoothing", type=int, default=1, help="Radio smoothing Chan–Vese (skimage)")
    parser.add_argument(
        "--chanvese-log-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loguea progreso aproximado por pasos internos del optimizador.",
    )
    parser.add_argument("--deeplab-device", choices=("cpu", "cuda"), default="cpu", help="Dispositivo inferencia DeepLab")
    parser.add_argument(
        "--deeplab-min-side",
        type=int,
        default=520,
        help="Lado corto mínimo al redimensionar antes de la red (mejor detalle, más costo).",
    )
    parser.add_argument(
        "--deeplab-class",
        type=int,
        default=-1,
        help="Índice canal PASCAL VOC (0..20); -1 = auto 'person' si existe en meta.",
    )
    parser.add_argument(
        "--unet-weights",
        type=Path,
        default=None,
        help="Checkpoint .pth state_dict compatible con MiniUNet 1 canal (obligatorio si --preprocess-mode unet).",
    )
    parser.add_argument("--unet-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--unet-threshold",
        type=float,
        default=0.5,
        help="Umbral sigmoid salida (0 = solo probabilidad suavizada).",
    )
    parser.add_argument(
        "--sam2-prompts",
        type=Path,
        default=None,
        help="JSON con input_boxes y/o input_points+input_labels (coords en píxeles de la imagen ya reescalada).",
    )
    parser.add_argument(
        "--sam2-model-id",
        type=str,
        default="facebook/sam2.1-hiera-tiny",
        help="Modelo Hugging Face SAM2.x (descarga pesos al primer uso).",
    )
    parser.add_argument("--sam2-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--sam2-multimask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="multimask_output del modelo (varias hipótesis por prompt).",
    )
    parser.add_argument("--sam2-mask-index", type=int, default=0, help="Índice máscara si multimask (ciclo según canales).")
    parser.add_argument("--score-version", choices=("v1", "v2"), default="v1", help="Version de metrica de score (v2: SSIM+reg)")
    parser.add_argument(
        "--sampling",
        choices=("grid", "sobol"),
        default="sobol",
        help="Muestreo de la capa densa blue-family (Sobol por defecto).",
    )
    parser.add_argument("--sobol-seed", type=int, default=42, help="Semilla Sobol (capa densa)")
    parser.add_argument(
        "--epsilon-mode",
        choices=("fixed", "adaptive"),
        default="adaptive",
        help="Epsilon de checkpoints en modo guiado con lotes.",
    )
    parser.add_argument(
        "--register",
        choices=("none", "affine", "homography"),
        default="none",
        help="Alineacion ECC del input al target antes del preprocess.",
    )
    args = parser.parse_args()
    guided_flags = ("--guided-explore", "--no-guided-explore")
    if not any(x in sys.argv for x in guided_flags) and args.n > 2000:
        args.guided_explore = True
        print(f"[INFO] guided-explore activado automaticamente (n={args.n} > 2000)", flush=True)
    if args.n > MAX_CANDIDATES_PER_RUN:
        print(
            f"Aviso: --n={args.n} supera el máximo por corrida ({MAX_CANDIDATES_PER_RUN:,}); "
            f"se acota a {MAX_CANDIDATES_PER_RUN:,}.",
            flush=True,
        )
        args.n = MAX_CANDIDATES_PER_RUN
    if args.n > 400_000:
        print(
            f"Aviso: --n={args.n} puede consumir mucha RAM, tiempo y disco; "
            "usa --max-side > 0 para explorar antes en baja resolución.",
            flush=True,
        )

    try:
        input_image = load_rgb(args.input)
        target_image_full = load_rgb(args.target)
    except FileNotFoundError as exc:
        print(f"No existe: {exc}", file=sys.stderr)
        return 2

    prepare_output_dir(args.out, args.resume)
    target_image = target_image_full.copy()
    if args.max_side > 0:
        target_image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
    input_resized = resize_to_target(input_image, target_image.size, max_side=0)

    rgb_u8 = np.array(input_resized)
    if args.register != "none":
        try:
            from laser_registration import align_input_to_target, save_registration_debug

            tgt_rgb = np.array(target_image.convert("RGB"))
            rgb_u8, reg_meta = align_input_to_target(rgb_u8, tgt_rgb, args.register)
            save_registration_debug(args.out, rgb_u8, reg_meta)
            print(f"[REG] ECC modo={args.register}; ver aligned_input.png y transform.json", flush=True)
        except Exception as exc:
            print(f"[REG] aviso: sin alineacion ({exc})", flush=True)
    base_gray = rgb_to_gray(rgb_u8)
    mask_debug = (args.out / "preprocess_mask.png") if args.save_preprocess_mask else None
    pfeather = max(0, int(args.preprocess_mask_feather))

    if args.preprocess_mode == "sam2" and args.sam2_prompts is None:
        print("SAM2 requiere --sam2-prompts ruta.json (ver runs/references/sam2_prompt_example.json).", file=sys.stderr)
        return 2

    if args.preprocess_mode == "sauvola":
        base_gray = sauvola_preprocess_gray(
            base_gray,
            args.sauvola_window,
            args.sauvola_k,
            args.sauvola_r,
            args.sauvola_blend,
        )
    elif args.preprocess_mode == "niblack":
        base_gray = niblack_preprocess_gray(base_gray, args.sauvola_window, args.sauvola_k, args.sauvola_blend)
    elif args.preprocess_mode == "grabcut":
        try:
            grect = parse_grabcut_rect(args.grabcut_rect, rgb_u8.shape[1], rgb_u8.shape[0])
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            base_gray = grabcut_preprocess_gray(rgb_u8, base_gray, grect, feather=pfeather, debug_path=mask_debug)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    elif args.preprocess_mode == "watershed":
        try:
            base_gray = watershed_preprocess_gray(base_gray)
        except ImportError as exc:
            print(f"Watershed requiere scikit-image: {exc}", file=sys.stderr)
            return 2
    elif args.preprocess_mode == "chanvese":
        base_gray = chan_vese_preprocess_gray(
            base_gray,
            args.chanvese_iter,
            args.chanvese_smoothing,
            args.chanvese_log_progress,
            feather=pfeather,
            debug_path=mask_debug,
        )
    elif args.preprocess_mode == "deeplab":
        try:
            dcls = None if args.deeplab_class < 0 else args.deeplab_class
            base_gray = deeplab_preprocess_gray(
                rgb_u8,
                base_gray,
                args.deeplab_device,
                args.deeplab_min_side,
                dcls,
                feather=pfeather,
                debug_path=mask_debug,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    elif args.preprocess_mode == "unet":
        if args.unet_weights is None:
            print("U-Net requiere --unet-weights ruta.pth (state_dict MiniUNet 1 canal).", file=sys.stderr)
            return 2
        try:
            base_gray = unet_preprocess_gray(
                base_gray,
                args.unet_weights,
                args.unet_device,
                args.unet_threshold,
                feather=pfeather,
                debug_path=mask_debug,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    elif args.preprocess_mode == "sam2":
        try:
            base_gray = sam2_preprocess_gray(
                rgb_u8,
                base_gray,
                args.sam2_prompts,
                args.sam2_model_id,
                args.sam2_device,
                args.sam2_multimask,
                args.sam2_mask_index,
                pfeather,
                mask_debug,
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.preprocess_mode != "none":
        preview = np.clip(base_gray, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(args.out / "preprocessed_gray.png", optimize=True)
        print(f"Preprocesado: modo={args.preprocess_mode} (vista preprocessed_gray.png)", flush=True)

    target_gray = rgb_to_gray(np.array(target_image))
    if detect_binary_target(target_gray):
        target_binary = np.where(target_gray > 127, 255, 0).astype(np.uint8)
        print("[TARGET] detectado binario; saltando Otsu", flush=True)
    else:
        target_threshold = otsu_threshold(target_gray)
        target_binary = np.where(target_gray >= target_threshold, 255, 0).astype(np.uint8)
    Image.fromarray(target_binary, mode="L").save(args.out / "target_binary.png", optimize=True)
    target_density = density_map(target_gray)
    target_edges = edge_map(target_gray)
    target_white_ratio = float(np.mean(target_binary == 255))

    if args.refine_db is not None:
        candidates = local_refine_candidates(args.refine_db, args.refine_top, args.n, args.refine_best_per_algorithm)
    elif args.from_db is not None:
        candidates = read_top_candidates(args.from_db, args.from_db_top, args.from_db_best_per_algorithm)
    elif args.focus_threshold > 0:
        candidates = focused_candidates(args.focus_threshold, args.n)
    else:
        candidates = build_candidates(
            base_gray,
            target_white_ratio,
            args.n,
            sampling=args.sampling,
            sobol_seed=args.sobol_seed,
        )
    if len(candidates) >= 250_000:
        print(f"Candidatos únicos a evaluar: {len(candidates)} (mezcla densa + otras capas)", flush=True)
    db = ensure_db(args.out / "match.sqlite")
    manifest_path = args.out / "match_manifest.jsonl"
    results: list[MatchResult] = []
    t_all = time.perf_counter()

    if args.guided_explore:
        worker_count = args.workers if args.workers > 0 else default_worker_count()
        print(
            f"Modo guiado: workers={worker_count} candidatos_plan={len(candidates)} max_evals={args.n} "
            f"dedupe={args.dedupe_param_hashes} plateau={args.plateau_detect} batch_early_stop={args.batch_early_stop}",
            flush=True,
        )
        with manifest_path.open("w", encoding="utf-8") as manifest:
            results = guided_run_evaluation(
                args,
                candidates,
                base_gray,
                target_gray,
                target_binary,
                target_density,
                target_edges,
                target_white_ratio,
                manifest,
                db,
                args.out,
            )
    else:
        worker_count = args.workers
        if worker_count <= 0:
            worker_count = default_worker_count()
        num_candidates = len(candidates)
        chunk = parallel_chunksize(num_candidates, worker_count)
        print(
            f"Paralelismo: {worker_count} proceso(s) · {num_candidates} candidatos · chunksize {chunk}",
            flush=True,
        )
        print(
            f"[CONFIG] plateau_detect={args.plateau_detect} window={args.plateau_window} std_max={args.plateau_std_max}",
            flush=True,
        )
        if args.plateau_detect and not args.guided_explore and worker_count > 1:
            print(
                "[PLATEAU] plateau_detect sin guided-explore y workers>1: "
                "reinicio por plateau solo esta cableado en modo guiado.",
                flush=True,
            )

        init_worker(base_gray, target_gray, target_binary, target_density, target_edges, str(args.score_version))

        with manifest_path.open("w", encoding="utf-8") as manifest:
            tasks = list(enumerate(candidates, start=1))
            if worker_count == 1:
                evaluated = map(evaluate_candidate_task, tasks)
            else:
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=init_worker,
                    initargs=(
                        base_gray,
                        target_gray,
                        target_binary,
                        target_density,
                        target_edges,
                        str(args.score_version),
                    ),
                )
                evaluated = executor.map(evaluate_candidate_task, tasks, chunksize=chunk)

            for idx, candidate, output, score, pixel_error, edge_error, white_ratio, elapsed in evaluated:
                filename = f"match_{idx:04d}.png"
                Image.fromarray(output, mode="L").save(args.out / filename, optimize=True)
                row = MatchResult(
                    id=idx,
                    **asdict(candidate),
                    score=score,
                    pixel_error=pixel_error,
                    edge_error=edge_error,
                    white_ratio=white_ratio,
                    target_white_ratio=target_white_ratio,
                    output_file=filename,
                    seconds=elapsed,
                )
                results.append(row)
                manifest.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                db.execute(
                    """
                    INSERT INTO matches (
                        algorithm, invert, threshold, contrast, brightness, gamma, autocontrast, sharpen,
                        score, pixel_error, edge_error, white_ratio, target_white_ratio, output_file, seconds, created
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        row.algorithm,
                        int(row.invert),
                        row.threshold,
                        row.contrast,
                        row.brightness,
                        row.gamma,
                        row.autocontrast,
                        row.sharpen,
                        row.score,
                        row.pixel_error,
                        row.edge_error,
                        row.white_ratio,
                        row.target_white_ratio,
                        row.output_file,
                        row.seconds,
                    ),
                )
                if idx % 50 == 0 or idx == len(candidates):
                    db.commit()
                    print(f"  {idx}/{len(candidates)} best={min(r.score for r in results):.4f}", flush=True)

            if worker_count != 1:
                executor.shutdown()

    db.commit()
    db.close()
    if not results:
        print("Sin resultados (lista de candidatos vacia o evaluacion cortada).", file=sys.stderr)
        return 2
    results.sort(key=lambda r: r.score)
    total = time.perf_counter() - t_all
    report_rows = results[: args.top_report]
    write_contact_sheet(args.out, report_rows, args.thumb_side, min(args.top_report, 40))
    write_html(args.out, report_rows, args.input.name, args.target.name, args.thumb_side)
    write_best_report(
        args.out,
        results,
        args.input.name,
        args.target.name,
        meta_extra={
            "n_evaluated": len(results),
            "wallclock_seconds": float(total),
            "score_version": str(args.score_version),
            "sampling": str(args.sampling),
        },
    )

    print(f"Listo: {args.out}")
    print(f"  SQLite: {args.out / 'match.sqlite'}")
    print(f"  Reporte: {args.out / 'index.html'}")
    print(f"  Mejor:  {args.out / results[0].output_file}  score={results[0].score:.6f}")
    print(f"  Best report: {args.out / 'best_report.md'}")
    print(f"  Hoja top: {args.out / 'contact_sheet.png'}")
    print(f"  Target binario: {args.out / 'target_binary.png'}")
    print(f"  Tiempo total: {total:.1f}s")
    print("\nTop candidatos:")
    for row in results[:10]:
        print(
            f"  {row.output_file} score={row.score:.4f} px={row.pixel_error:.4f} edge={row.edge_error:.4f} "
            f"{row.algorithm} inv={int(row.invert)} thr={row.threshold} c={row.contrast:.2f} "
            f"b={row.brightness:+.0f} g={row.gamma:.2f} sharp={row.sharpen:.0f} white={row.white_ratio:.3f}"
        )
    try:
        from scripts.meta.recorder import record_experiment

        record_experiment(args.out, args, _REPO_ROOT)
    except Exception as exc:
        print(f"[META] recorder: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
