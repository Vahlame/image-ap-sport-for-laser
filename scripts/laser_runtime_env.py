"""Variables de entorno efectivas para procesos del repo (LPIPS / score v4 / CUDA / HF).

Garantiza que LASER_LPIPS_DEVICE nunca quede vacío: sin definir → ``auto`` (usa CUDA si PyTorch lo tiene).
En GPUs pequeñas (≤ ~7.5 GiB reportados) limita por defecto la reserva PyTorch a ~5.5 GiB por proceso
(``LASER_CUDA_MEMORY_CAP_GIB`` / ``apply_cuda_process_memory_cap``).
``sync_hf_hub_token_env`` alinea ``HF_TOKEN`` y ``HUGGING_FACE_HUB_TOKEN`` para SAM2 / Hugging Face Hub.
Los wrappers que lanzan ``laser_target_match`` por subprocess deben usar ``child_process_env()``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

ENV_LPIPS_DEVICE = "LASER_LPIPS_DEVICE"
# Techo de VRAM por **proceso** (GiB) vía `torch.cuda.set_per_process_memory_fraction`.
# Si no está definido y la GPU 0 reporta ≤7.5 GiB totales, se usa 5.5 GiB por defecto.
ENV_CUDA_MEMORY_CAP_GIB = "LASER_CUDA_MEMORY_CAP_GIB"
# Hugging Face Hub (SAM2, pesos): https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_TOKEN_LEGACY = "HUGGING_FACE_HUB_TOKEN"


def sync_hf_hub_token_env(target: dict[str, str] | None = None) -> None:
    """
    Hugging Face acepta ``HF_TOKEN`` o ``HUGGING_FACE_HUB_TOKEN``; duplica el valor
    para que librerías que lean cualquiera de los dos funcionen igual.
    """
    env = os.environ if target is None else target
    t = str(env.get(ENV_HF_TOKEN, "") or "").strip()
    h = str(env.get(ENV_HF_TOKEN_LEGACY, "") or "").strip()
    if t and not h:
        env[ENV_HF_TOKEN_LEGACY] = t
    elif h and not t:
        env[ENV_HF_TOKEN] = h


def apply_cuda_process_memory_cap(*, quiet: bool = False) -> None:
    """
    Limita la fracción de VRAM que PyTorch puede reservar en este proceso (CUDA).

    - ``LASER_CUDA_MEMORY_CAP_GIB=off`` o ``0``: no aplica límite.
    - Valor numérico (ej. ``5.5``): techo en gibibytes respecto al total reportado por la GPU.
    - Sin variable y VRAM total ≤ 7.5 GiB: aplica **5.5 GiB** por defecto (placas ~6 GB).
    - Sin variable y VRAM > 7.5 GiB: no hace nada (definí la variable si querés tope en GPUs grandes).

    Debe llamarse **antes** del primer tensor en GPU (main y cada worker hijo).
    """
    raw = str(os.environ.get(ENV_CUDA_MEMORY_CAP_GIB, "") or "").strip().lower()
    if raw in ("0", "off", "no", "false"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return
    total_gib = total_bytes / (1024.0**3)
    if raw:
        try:
            cap_gib = float(raw)
        except ValueError:
            return
    else:
        if total_gib > 7.5:
            return
        cap_gib = 5.5
    if cap_gib <= 0:
        return
    frac = min(0.99, cap_gib / max(total_gib, 1e-6))
    try:
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
    except Exception:
        return
    if not quiet:
        print(
            f"[CONFIG] CUDA: techo VRAM proceso ~{cap_gib:.2f} GiB "
            f"(fraccion {frac:.3f} del total ~{total_gib:.2f} GiB; env {ENV_CUDA_MEMORY_CAP_GIB} o off para desactivar)",
            flush=True,
        )


def resolve_torch_device_flag(raw: str) -> str:
    """
    Normaliza flags de CLI / env para inferencia PyTorch (DeepLab, U-Net, SAM2, etc.).

    - ``cpu`` -> ``cpu``
    - ``cuda`` / ``gpu`` -> ``cuda`` (el caller debe comprobar disponibilidad si hace falta)
    - ``auto`` -> ``cuda`` si ``torch.cuda.is_available()``; si no hay torch o CUDA, ``cpu``
    """
    v = str(raw or "cpu").strip().lower()
    if v in ("gpu", "cuda"):
        return "cuda"
    if v == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return "cpu"


def infer_v4_max_gpu_workers_cap() -> int:
    """
    Tope conservador de procesos paralelos cuando cada uno carga LPIPS en la misma GPU.

    Se usa solo si **no** está definido ``LASER_V4_MAX_GPU_WORKERS`` (entero).
    Evita OOM y contienda típica (muchas copias de Alex en VRAM).
    """
    try:
        import torch
    except ImportError:
        return 2
    if not torch.cuda.is_available():
        return 2
    try:
        mem_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 2
    mem_gb = mem_bytes / (1024.0**3)
    # GPUs ~4–6 GB (marketing) suelen reportar un poco menos en PyTorch; un solo
    # proceso con Alex+LPIPS + activaciones ya ocupa buena parte de la VRAM.
    if mem_gb < 7.5:
        return 1
    if mem_gb < 18.0:
        return 2
    if mem_gb < 30.0:
        return 3
    return min(4, max(3, int(mem_gb // 12)))


def lpips_device_mode() -> str:
    """
    Modo normalizado: ``auto`` | ``cuda`` | ``cpu``.

    ``gpu`` se trata como ``cuda``. Vacío o ausente → ``auto``.
    """
    raw = str(os.environ.get(ENV_LPIPS_DEVICE, "") or "").strip().lower()
    if raw in ("", "default"):
        return "auto"
    if raw in ("gpu", "cuda"):
        return "cuda"
    if raw == "cpu":
        return "cpu"
    if raw == "auto":
        return "auto"
    raise ValueError(f"{ENV_LPIPS_DEVICE} invalido: {raw!r} (use auto|cuda|gpu|cpu)")


def apply_lpips_default_process_env() -> None:
    """Si la variable no está definida o está vacía, fija ``auto`` en este proceso."""
    if not str(os.environ.get(ENV_LPIPS_DEVICE, "") or "").strip():
        os.environ[ENV_LPIPS_DEVICE] = "auto"


def coerce_lpips_env_if_cuda_unavailable() -> None:
    """
    Si el usuario forzó CUDA pero este intérprete no tiene GPU disponible, baja a ``cpu``.

    Evita que procesos hijos (multiprocessing) fallen al heredar ``LASER_LPIPS_DEVICE=cuda``
    de la shell cuando PyTorch es CPU-only.
    """
    raw = str(os.environ.get(ENV_LPIPS_DEVICE, "") or "").strip().lower()
    if raw not in ("cuda", "gpu"):
        return
    try:
        import torch
    except ImportError:
        os.environ[ENV_LPIPS_DEVICE] = "cpu"
        return
    if not torch.cuda.is_available():
        os.environ[ENV_LPIPS_DEVICE] = "cpu"


def child_process_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    Copia del entorno para ``subprocess``: ``LASER_LPIPS_DEVICE`` nunca vacío;
    ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` se sincronizan para SAM2 y descargas HF.
    Usar en todos los lanzamientos de ``laser_target_match.py`` (y scripts equivalentes).
    """
    env = dict(os.environ if environ is None else environ)
    sync_hf_hub_token_env(env)
    if not str(env.get(ENV_LPIPS_DEVICE, "") or "").strip():
        env[ENV_LPIPS_DEVICE] = "auto"
    # Evita UnicodeEncodeError en prints del hijo cuando el padre captura stdout/stderr (cp1252).
    if sys.platform == "win32":
        env.setdefault("PYTHONIOENCODING", "utf-8")
    return env
