"""Alineacion ligera input-target con ECC (OpenCV)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

WarpMode = Literal["none", "affine", "homography"]


def align_input_to_target(
    input_rgb: np.ndarray,
    target_rgb: np.ndarray,
    warp_mode: WarpMode = "affine",
    max_iter: int = 300,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    Alinea `input_rgb` hacia `target_rgb` con findTransformECC.

    Args:
        input_rgb: uint8 HxWx3 RGB.
        target_rgb: uint8 mismo tamano que input tras resize previo.
        warp_mode: affine u homography; none devuelve copia sin cambio.
        max_iter: max iteraciones ECC.
        eps: criterio de convergencia ECC.

    Returns:
        Tupla (imagen alineada uint8 RGB, dict serializable con matriz y meta).

    Raises:
        RuntimeError: si OpenCV no esta disponible o ECC falla.
    """
    if warp_mode == "none":
        return input_rgb.copy(), {"mode": "none", "matrix": None}
    if cv2 is None:
        raise RuntimeError("[REG] OpenCV requerido para --register; pip install opencv-python-headless")

    if input_rgb.shape != target_rgb.shape:
        raise ValueError("[REG] input y target deben tener la misma forma")

    in_g = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tg_g = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    in_g = cv2.GaussianBlur(in_g, (5, 5), 0)
    tg_g = cv2.GaussianBlur(tg_g, (5, 5), 0)

    h, w = in_g.shape
    if warp_mode == "affine":
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        motion = cv2.MOTION_AFFINE
    elif warp_mode == "homography":
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        motion = cv2.MOTION_HOMOGRAPHY
    else:
        raise ValueError(f"[REG] warp_mode invalido: {warp_mode}")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iter), float(eps))
    try:
        _cc, warp_matrix = cv2.findTransformECC(
            tg_g,
            in_g,
            warp_matrix,
            motion,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error as exc:
        raise RuntimeError(f"[REG] findTransformECC fallo: {exc}") from exc

    if warp_mode == "affine":
        warped = cv2.warpAffine(
            input_rgb,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        meta: dict = {"mode": "affine", "matrix": warp_matrix.tolist()}
    else:
        warped = cv2.warpPerspective(
            input_rgb,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        meta = {"mode": "homography", "matrix": warp_matrix.tolist()}
    return warped.astype(np.uint8), meta


def save_registration_debug(out_dir: Path, aligned_rgb: np.ndarray, meta: dict) -> None:
    """Guarda `aligned_input.png` y `transform.json` bajo out_dir."""
    if cv2 is None:
        raise RuntimeError("[REG] OpenCV requerido para guardar aligned_input.png")
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "aligned_input.png"), bgr)
    (out_dir / "transform.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
