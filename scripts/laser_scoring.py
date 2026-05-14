"""Metricas de scoring v1 (legacy) y v2 (SSIM + regularizacion de parametros)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Magnitud de gradiente normalizada 0..1 (misma semantica que laser_target_match)."""
    y_grad, x_grad = np.gradient(gray.astype(np.float64) / 255.0)
    mag = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    p95 = np.percentile(mag, 95)
    if p95 > 0:
        mag = mag / p95
    return np.clip(mag, 0.0, 1.0)


def density_map(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """Downscale bilineal para comparar densidad local."""
    h, w = gray.shape
    small_size = (max(1, w // scale), max(1, h // scale))
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    return np.array(image.resize(small_size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0


def score_candidate_v1(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
) -> tuple[float, float, float, float]:
    """Scoring original (densidad dominante)."""
    out_norm = out.astype(np.float64) / 255.0
    target_norm = target_binary.astype(np.float64) / 255.0
    pixel_error = float(np.mean((out_norm - target_norm) ** 2))
    density_error = float(np.mean((density_map(out) - target_density) ** 2))
    edge_error = float(np.mean(np.abs(edge_map(out) - target_edges)))
    white_ratio = float(np.mean(out == 255))
    target_white_ratio = float(np.mean(target_binary == 255))
    ratio_error = abs(white_ratio - target_white_ratio)
    raw_tone_error = float(abs(np.mean(out_norm) - np.mean(target_gray / 255.0)))
    score = (
        0.52 * density_error
        + 0.24 * edge_error
        + 0.12 * pixel_error
        + 0.08 * ratio_error
        + 0.04 * raw_tone_error
    )
    return score, pixel_error, edge_error, white_ratio


def score_candidate_v2(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any,
) -> tuple[float, float, float, float]:
    """
    Scoring v2: SSIM sobre luminancia continua + terminos legacy con pesos rebalanceados
    y regularizacion suave contra contrastes bajos, brillo extremo y sharpen alto.

    Returns:
        (score, pixel_error, edge_error, white_ratio) — mismas 4 metricas auxiliares que v1
        para compatibilidad con SQLite/reportes; `score` es la metrica objetivo v2.
    """
    out_norm = out.astype(np.float64) / 255.0
    tgt_cont = np.clip(target_gray.astype(np.float64) / 255.0, 0.0, 1.0)
    ssim_term = 1.0 - float(
        structural_similarity(
            out_norm,
            tgt_cont,
            data_range=1.0,
        )
    )

    target_norm_bin = target_binary.astype(np.float64) / 255.0
    pixel_error = float(np.mean((out_norm - target_norm_bin) ** 2))
    density_error = float(np.mean((density_map(out) - target_density) ** 2))
    edge_error = float(np.mean(np.abs(edge_map(out) - target_edges)))
    white_ratio = float(np.mean(out == 255))
    target_white_ratio = float(np.mean(target_binary == 255))
    ratio_error = abs(white_ratio - target_white_ratio)

    c = float(candidate.contrast)
    b = float(candidate.brightness)
    sh = float(candidate.sharpen)
    reg = (
        0.20 * max(0.0, 0.70 - c)
        + 0.15 * (max(0.0, abs(b) - 25.0) / 40.0)
        + 0.10 * (max(0.0, sh - 100.0) / 100.0)
    )

    score = (
        0.40 * ssim_term
        + 0.20 * pixel_error
        + 0.15 * edge_error
        + 0.10 * density_error
        + 0.05 * ratio_error
        + 0.10 * reg
    )
    return score, pixel_error, edge_error, white_ratio


ScoreVersion = Literal["v1", "v2"]


def score_candidate_dispatch(
    version: str,
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any | None = None,
) -> tuple[float, float, float, float]:
    """Enruta a v1 o v2. v2 requiere `candidate` para la regularizacion."""
    if version == "v1":
        return score_candidate_v1(out, target_gray, target_binary, target_density, target_edges)
    if version == "v2":
        if candidate is None:
            raise ValueError("[SCORING] score-version v2 requiere candidate")
        return score_candidate_v2(out, target_gray, target_binary, target_density, target_edges, candidate)
    raise ValueError(f"[SCORING] version desconocida: {version!r}")
