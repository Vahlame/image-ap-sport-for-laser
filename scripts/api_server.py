#!/usr/bin/env python3
"""
FastAPI backend para el wizard SvelteKit.

Expone el motor Python (laser_target_match + laser_physics + laser_scoring) a la UI web
con una API REST minima. Pensado para correr local (uvicorn) junto al wizard Vite.

Endpoints:
- GET  /api/health         liveness + estado modelo
- GET  /api/materials      builtins + JSON custom de presets/materials/
- GET  /api/algorithms     ALL_RENDER_ALGORITHMS agrupado por familia
- POST /api/preview        procesa con max-side=400 (rapido, <2s tipico)
- POST /api/process        procesa full-res; devuelve PNG 1-bit

Diseno:
- Stateless: cada request es independiente.
- LPIPS modelo lazy-loaded (compartido entre requests via singleton de laser_scoring).
- CORS abierto para dev (localhost:5173 y :4173).
- Sin auth: pensado para uso local. Si se expone en red, ANTES agregar auth.

Uso:
    pip install -e ".[api,perceptual]"
    uvicorn scripts.api_server:app --reload --host 127.0.0.1 --port 18765

Acceso desde el wizard:
    fetch('http://127.0.0.1:18765/api/process', { method: 'POST', body: formData })
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import laser_target_match as ltm
import laser_scoring
import laser_physics
import laser_simulator
import laser_presets
from laser_jobs import REGISTRY, JobState

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise SystemExit(
        "Falta dependencia API. Instalar con: pip install -e \".[api]\""
    ) from exc


# ---------------------------------------------------------------------------
# Modelos pydantic
# ---------------------------------------------------------------------------


class ProcessParams(BaseModel):
    """Parametros del pipeline. Defaults sensatos para acrilico back-engrave."""

    preset: str = Field(
        default="",
        description="Si se pasa nombre de preset (e.g. 'photo_general', 'auto'), aplica esos "
        "params como base; campos explicitos en este JSON SOBREESCRIBEN el preset."
    )
    quality_mode: str = Field(
        default="fast",
        description="'fast' (1 render) o 'best' (sobol search ~24 candidatos + v5 scoring). "
        "/api/process default 'best'; /api/preview default 'fast'."
    )
    material: str = Field(default="", description="Nombre MaterialProfile o vacio")
    output_mm_short: float = Field(default=0.0, ge=0, description="Lado corto fisico en mm (0=no escalar USM)")
    output_dpi: int = Field(default=0, ge=0, le=2400, description="DPI del grabado final (0=no aplicar)")
    algorithm: str = Field(default="floyd", description="Algoritmo de dither")
    threshold: int = Field(default=83, ge=1, le=254)
    contrast: float = Field(default=0.55, ge=0.1, le=3.0)
    brightness: float = Field(default=25.0, ge=-100, le=100)
    gamma: float = Field(default=1.35, ge=0.3, le=3.0)
    autocontrast: float = Field(default=0.0, ge=0, le=20)
    sharpen: float = Field(default=40.0, ge=0, le=300)
    sharpen_radius_mm: float = Field(default=0.10, ge=0.01, le=2.0)
    invert: bool = Field(default=True)
    preprocess_mode: str = Field(default="sauvola", description="Pre-CV: none|sauvola|niblack|clahe|sauvola_clahe|grabcut|chanvese|sam2")
    max_side: int = Field(default=0, ge=0, le=8000, description="0 = no resize (full-res). Preview pasa 400.")
    simplify_plain_regions: bool = Field(
        default=True,
        description="Si True (default), detecta zonas localmente uniformes (cielo, fondos planos) "
        "y las clampea a blanco/negro puro antes del dither. Reduce ruido moteado en el grabado "
        "y ahorra pulsos del láser. Desactivar para preservar texturas sutiles en fondos.",
    )
    s_curve_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.5,
        description="Curva en S tonal (0=sin, 0.5=suave, 1.0=agresiva). Aclara midtones y "
        "oscurece sombras → look fotorrealista profesional. Técnica estándar Photoshop/PhotoGrav.",
    )
    local_contrast_amount: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Local contrast enhancement % (a.k.a. 'Clarity'). Unsharp Mask con radius "
        "grande (60px) + amount bajo (default 0=sin, 5-20 típico). Aumenta 'punch' fotográfico "
        "sin amplificar ruido como CLAHE.",
    )
    auto_mirror_back_engrave: bool = Field(
        default=True,
        description="Si True (default) y material termina en '_back_engrave', voltea el PNG "
        "final horizontalmente. PhotoGrav lo hace automáticamente porque al grabar en la cara "
        "POSTERIOR del acrílico, hay que invertir para que se vea correcto desde el frente.",
    )
    score_version: str = Field(
        default="v5",
        description="Métrica de calidad para HQ refine: 'v5' (no-reference, CPU rápido, default) "
        "o 'v4' (LPIPS perceptual, requiere torch+lpips, usa GPU si está disponible). "
        "v4 genera auto-target binario por threshold-50 del gris.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cuda_available: bool
    repo_root: str


class MaterialInfo(BaseModel):
    name: str
    spot_mm: float
    default_dpi: int
    tone_response: str
    power_pct_range: tuple[float, float]
    notes: str
    source: str  # "builtin" | "custom"
    # Nuevos v1.3 (opcionales para retrocompat con perfiles custom JSON sin estos campos)
    speed_mm_s_range: tuple[float, float] = (200.0, 600.0)
    pass_through: bool = True
    mirror_x_required: bool = False
    lightburn_invert: bool = False
    focus_mm: float = 0.0
    machine_compat: str = ""


class RecommendedSettings(BaseModel):
    material: str
    machine_compat: str
    spot_mm: float
    focus_mm: float
    dpi: int
    interval_mm: float
    power_pct_min: float
    power_pct_max: float
    speed_mm_s_min: float
    speed_mm_s_max: float
    pass_through: bool
    mirror_x_required: bool
    lightburn_invert: bool
    tone_response: str
    notes: str


class AlgorithmGroup(BaseModel):
    family: str
    algorithms: list[str]


# ---------------------------------------------------------------------------
# Helpers de procesamiento
# ---------------------------------------------------------------------------


MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB max upload — protección contra DoS/OOM


def _load_image_from_bytes(data: bytes) -> Image.Image:
    """Decodifica bytes a PIL Image RGB (raise HTTPException si invalido).

    v2.1 — Validaciones:
    - Tamaño bytes <= MAX_UPLOAD_BYTES (100MB) → 413 Payload Too Large.
    - Dimensiones 16x16 .. 8000x8000.
    - PIL.Image.open con context manager (cierra recursos en error path).
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"imagen demasiado grande ({len(data)//(1024*1024)} MB) — máximo {MAX_UPLOAD_BYTES//(1024*1024)} MB",
        )
    if len(data) < 64:
        raise HTTPException(
            status_code=400, detail=f"imagen invalida — vacía o corrupta ({len(data)} bytes)"
        )
    try:
        # Context manager para que PIL cierre recursos si convert() falla
        with Image.open(io.BytesIO(data)) as raw:
            img = raw.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"imagen invalida: {exc}") from exc
    if img.size[0] < 16 or img.size[1] < 16:
        raise HTTPException(status_code=400, detail=f"imagen muy chica: {img.size}")
    if img.size[0] > 8000 or img.size[1] > 8000:
        raise HTTPException(status_code=400, detail=f"imagen muy grande: {img.size}")
    return img


def _apply_preprocess(base_gray: np.ndarray, mode: str) -> np.ndarray:
    """Aplica preprocess seleccionado. Solo modos no-CV / livianos en este path."""
    if mode == "none":
        return base_gray
    if mode == "sauvola":
        return ltm.sauvola_preprocess_gray(base_gray, window=15, k=0.15, R=128.0, blend=0.35)
    if mode == "niblack":
        return ltm.niblack_preprocess_gray(base_gray, window=15, k=-0.2, blend=0.35)
    if mode == "clahe":
        # CLAHE adaptativo: usa blend dinámico según std de la imagen para evitar
        # over-enhance de bokehs (std bajo = imagen suave/desenfocada → blend menor;
        # std alto = imagen con detalle distribuido → blend completo).
        std = float(base_gray.std())
        if std < 50.0:
            # Imagen suave (bokeh, lighting controlado): CLAHE leve
            blend = 0.30
        elif std < 70.0:
            blend = 0.45
        else:
            blend = 0.60  # imagen con mucho detalle: CLAHE completo
        return ltm.clahe_preprocess_gray(base_gray, clip_limit=2.5, tile_size=8, blend=blend)
    if mode == "sauvola_clahe":
        # Pipeline 2-pass: primero CLAHE para revelar detalles, después Sauvola para
        # contraste local refinado. Mejor para imágenes con detalles importantes
        # en zonas extremas (claras o oscuras).
        clahe = ltm.clahe_preprocess_gray(base_gray, clip_limit=2.0, tile_size=8, blend=0.5)
        return ltm.sauvola_preprocess_gray(clahe, window=15, k=0.15, R=128.0, blend=0.30)
    # grabcut/chanvese/sam2 requieren mas args; los exponemos como preprocess avanzado en CLI
    raise HTTPException(
        status_code=400,
        detail=f"preprocess_mode '{mode}' no soportado en API "
        "(usar CLI o uno de: none, sauvola, niblack, clahe, sauvola_clahe)",
    )


def _resolve_material_lut(material_name: str, presets_dir: Path | None) -> tuple[Any | None, Any | None]:
    """Devuelve (profile, lut_callable) o (None, None) si material vacio."""
    if not material_name:
        return None, None
    try:
        profile = laser_physics.load_material_profile(material_name, presets_dir=presets_dir)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=f"material '{material_name}': {exc}") from exc
    return profile, profile.lut()


def _resolve_sharpen_radius(
    output_mm_short: float, output_dpi: int, ranking_short: int, radius_mm: float
) -> float:
    """Escala radius al output fisico (regla 3) o devuelve default 1.2 si faltan datos."""
    if output_mm_short <= 0 or output_dpi <= 0:
        return 1.2
    return laser_physics.scaled_unsharp_radius(
        ranking_pixels_short_side=ranking_short,
        output_mm_short_side=float(output_mm_short),
        output_dpi=int(output_dpi),
        radius_mm=float(radius_mm),
    )


def _resolve_preset_overrides(params: ProcessParams, img_rgb: Image.Image) -> tuple[ProcessParams, str, str]:
    """
    Si params.preset != '', aplica el preset como base y respeta los campos del request
    como overrides explícitos.

    Convención: si en `params` un campo coincide con el DEFAULT del schema, se considera
    "no especificado por el usuario" y se reemplaza por el del preset. Si el usuario
    cambió ese campo, su valor manda.

    Si preset == 'auto', corre el detector sobre la imagen y elige.

    Returns: (params_efectivos, preset_name_aplicado, motivo_recomendacion)
    """
    preset_name = (params.preset or "").strip().lower()
    if not preset_name:
        return params, "", ""

    detector_reason = ""
    if preset_name == "auto":
        rgb_arr = np.array(img_rgb.convert("RGB"))
        # Pasar material para que el detector use el preset acrilico-especifico si aplica
        rec = laser_presets.recommend_preset(rgb_arr, material=params.material)
        preset_name = rec.preset_name
        detector_reason = rec.reason

    try:
        preset = laser_presets.get_preset(preset_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Defaults del schema para detectar "no tocado por usuario"
    defaults = ProcessParams().model_dump()
    user_dump = params.model_dump()
    preset_params = preset.as_param_dict()

    merged = dict(user_dump)
    for key, preset_val in preset_params.items():
        if key not in user_dump:
            continue
        # Si el usuario dejó el default, usar el del preset
        if user_dump[key] == defaults[key]:
            merged[key] = preset_val

    # Material: si el preset lo sugiere y el usuario no especificó, usarlo
    if not user_dump.get("material") and preset.suggested_material:
        merged["material"] = preset.suggested_material

    return ProcessParams(**merged), preset_name, detector_reason


def _hq_refine(
    base_gray: np.ndarray,
    base_params: ProcessParams,
    progress_cb: "Any | None" = None,
    cancel_check: "Any | None" = None,
) -> tuple[np.ndarray, dict, ltm.Candidate]:
    """
    HQ refinement: corre un sobol-search local (~24 variantes alrededor de los params dados)
    y devuelve el mejor por score v5 (no-reference: HVS-MSE + spectral radial + tone post-LUT).

    Args:
        base_gray: gris preprocesado.
        base_params: parámetros base; variantes se generan alrededor.
        progress_cb: callable opcional `(current, total, best_score, elapsed_seconds) -> None`
            llamado tras evaluar cada candidato. Sirve para reportar progreso al cliente.
        cancel_check: callable opcional `() -> bool` consultado entre candidatos. Si devuelve
            True, aborta y devuelve el mejor encontrado hasta el momento.

    Returns: (best_binary_output, debug_dict, winning_candidate)
    """
    from scipy.stats import qmc

    base_cand = ltm.Candidate(
        algorithm=base_params.algorithm,
        invert=base_params.invert,
        threshold=int(base_params.threshold),
        contrast=float(base_params.contrast),
        brightness=float(base_params.brightness),
        gamma=float(base_params.gamma),
        autocontrast=float(base_params.autocontrast),
        sharpen=float(base_params.sharpen),
    )

    # Sobol 5D, 24 puntos (no-warning con power_of_2=False; 24 elegidos para balance velocidad/cobertura).
    sobol = qmc.Sobol(d=5, seed=2026, scramble=True).random(24)
    variants: list[ltm.Candidate] = [base_cand]
    for s in sobol:
        thr = int(np.clip(base_cand.threshold + (s[0] - 0.5) * 32, 30, 220))
        c = float(np.clip(base_cand.contrast + (s[1] - 0.5) * 0.40, 0.4, 2.2))
        b = float(np.clip(base_cand.brightness + (s[2] - 0.5) * 24, -50, 50))
        g = float(np.clip(base_cand.gamma + (s[3] - 0.5) * 0.40, 0.5, 2.2))
        sh = float(np.clip(base_cand.sharpen + (s[4] - 0.5) * 60, 0, 250))
        variants.append(ltm.Candidate(
            algorithm=base_cand.algorithm, invert=base_cand.invert,
            threshold=thr, contrast=c, brightness=b, gamma=g,
            autocontrast=base_cand.autocontrast, sharpen=sh,
        ))

    # Targets para el scorer. v5 (no-ref) sólo usa target_gray; v4 (LPIPS perceptual)
    # necesita target_binary + density + edges. Como no tenemos un PNG "ideal" de referencia
    # en el flujo Express, generamos un mock target_binary por threshold-128 del gris:
    # es suficiente para que v4 corra y el LPIPS evalúe similitud perceptual entre el
    # output y un proxy razonable del gris pretendido.
    score_version = (base_params.score_version or "v5").lower()
    target_gray_v5 = base_gray.astype(np.float64)
    if score_version == "v4":
        target_binary = np.where(base_gray >= 128, 255, 0).astype(np.uint8)
        # density a 1/4 resolución (convención v1-v4)
        h_q, w_q = max(1, base_gray.shape[0] // 4), max(1, base_gray.shape[1] // 4)
        from PIL import Image as _PILImage  # local import (PIL ya está arriba pero esto evita shadowing)
        bin_pil = _PILImage.fromarray(target_binary, mode="L").resize((w_q, h_q), _PILImage.Resampling.LANCZOS)
        target_density = (np.array(bin_pil, dtype=np.float64) / 255.0)
        # edges via sobel sobre el gris
        from scipy import ndimage as _ndi
        gx = _ndi.sobel(base_gray.astype(np.float64), axis=1)
        gy = _ndi.sobel(base_gray.astype(np.float64), axis=0)
        target_edges = np.sqrt(gx * gx + gy * gy)
    else:
        target_binary = np.zeros_like(base_gray, dtype=np.uint8)
        target_density = np.zeros((max(1, base_gray.shape[0] // 4), max(1, base_gray.shape[1] // 4)), dtype=np.float64)
        target_edges = np.zeros_like(base_gray, dtype=np.float64)

    # Min improvement: el HQ refine sólo cambia el baseline si el nuevo candidato es
    # ≥ MIN_IMPROVEMENT mejor (5% relativo). Sin esto, refine puede moverse a un
    # candidato sutilmente mejor en score pero con diferencias visuales adversas
    # (caso observado en rally car: refine elegía contraste alto que comía detalles).
    MIN_IMPROVEMENT_RATIO = 0.05

    best_score = float("inf")
    best_out: np.ndarray | None = None
    best_cand: ltm.Candidate | None = None
    baseline_score = float("inf")
    scores_log: list[tuple[float, ltm.Candidate]] = []
    cancelled = False
    t0 = time.perf_counter()
    total = len(variants)
    for i, cand in enumerate(variants, start=1):
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        out = ltm.render_candidate(base_gray, cand)
        score, *_ = laser_scoring.score_candidate_dispatch(
            score_version, out, target_gray_v5, target_binary, target_density, target_edges, cand,
        )
        scores_log.append((float(score), cand))
        if i == 1:
            # Primer candidato = baseline (params del preset, sin variar)
            baseline_score = float(score)
            best_score = float(score)
            best_out = out
            best_cand = cand
        elif score < best_score:
            # Aceptar refinement sólo si mejora MEANINGFUL (> MIN_IMPROVEMENT_RATIO%)
            relative_improvement = (baseline_score - score) / (abs(baseline_score) + 1e-9)
            if relative_improvement >= MIN_IMPROVEMENT_RATIO:
                best_score = score
                best_out = out
                best_cand = cand
        if progress_cb is not None:
            try:
                progress_cb(i, total, float(best_score), time.perf_counter() - t0)
            except Exception as cb_exc:
                # Errores en progress_cb NO deben abortar el procesamiento,
                # pero sí queremos visibilidad (no silent swallow). Log + continúa.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "progress_cb raised at candidate %d/%d: %s", i, total, cb_exc
                )
    elapsed = time.perf_counter() - t0

    assert best_out is not None and best_cand is not None
    debug = {
        "candidates_evaluated": len(scores_log),
        "candidates_total": total,
        "score_version": score_version,
        "best_score_v5": best_score,  # nombre legacy para clientes existentes; semántica = score elegido
        "baseline_score_v5": scores_log[0][0] if scores_log else best_score,
        "improvement_v5": (scores_log[0][0] - best_score) if scores_log else 0.0,
        "refine_seconds": elapsed,
        "cancelled": cancelled,
    }
    return best_out, debug, best_cand


def _process_image(
    img_rgb: Image.Image,
    params: ProcessParams,
    progress_cb: "Any | None" = None,
    cancel_check: "Any | None" = None,
) -> tuple[np.ndarray, dict]:
    """
    Pipeline completo: resize -> preprocess -> apply LUT material -> render candidate.
    Devuelve (binario uint8, metadata dict).

    Si `progress_cb` y `cancel_check` se pasan, se propagan al HQ refine para reportar
    progreso de cada candidato evaluado y permitir cancelación temprana.
    """
    # Resize si max_side > 0
    w, h = img_rgb.size
    if params.max_side > 0 and max(w, h) > params.max_side:
        scale = params.max_side / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img_rgb = img_rgb.resize(new_size, Image.Resampling.LANCZOS)

    base_gray = ltm.rgb_to_gray(np.array(img_rgb))
    base_gray = _apply_preprocess(base_gray, params.preprocess_mode)

    # Aplicar LUT material si corresponde
    profile, lut_fn = _resolve_material_lut(
        params.material, presets_dir=_REPO_ROOT / "presets" / "materials"
    )
    if lut_fn is not None:
        base_gray = lut_fn(base_gray.astype(np.uint8)).astype(np.float64)

    # S-curve tonal: aclara midtones, oscurece sombras (workflow profesional).
    # Aplicar ANTES del local_contrast para que actúe sobre tonos puros.
    if params.s_curve_strength > 0:
        base_gray = ltm.apply_s_curve(base_gray, strength=params.s_curve_strength)

    # Local contrast enhancement: "Clarity"/PhotoGrav. Aumenta punch mid-frecuencia
    # sin amplificar ruido como CLAHE.
    if params.local_contrast_amount > 0:
        base_gray = ltm.apply_local_contrast(base_gray, radius_px=60.0, amount_pct=params.local_contrast_amount)

    # Plain region simplification: elimina dither moteado en zonas uniformes (cielo,
    # fondos blancos). Aplica DESPUÉS de LUT para que las zonas extremas post-LUT
    # se clampeen correctamente.
    if params.simplify_plain_regions:
        base_gray = ltm.plain_region_simplification(base_gray)

    # Configurar sharpen global (worker globals de ltm)
    ranking_short = int(min(base_gray.shape))
    resolved_radius = _resolve_sharpen_radius(
        params.output_mm_short, params.output_dpi, ranking_short, params.sharpen_radius_mm
    )
    ltm._WORK_SHARPEN_RADIUS = resolved_radius
    ltm._WORK_PPD = 64.0

    # Validar algoritmo
    if params.algorithm not in ltm.ALL_RENDER_ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"algorithm '{params.algorithm}' no soportado. Lista en /api/algorithms.",
        )

    quality_mode = (params.quality_mode or "fast").lower()
    refine_debug: dict | None = None
    if quality_mode == "best":
        t0 = time.perf_counter()
        out, refine_debug, winning_cand = _hq_refine(
            base_gray, params, progress_cb=progress_cb, cancel_check=cancel_check
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Reportar los params reales que ganaron
        winning_algorithm = winning_cand.algorithm
        winning_threshold = winning_cand.threshold
    else:
        cand = ltm.Candidate(
            algorithm=params.algorithm,
            invert=params.invert,
            threshold=params.threshold,
            contrast=params.contrast,
            brightness=params.brightness,
            gamma=params.gamma,
            autocontrast=params.autocontrast,
            sharpen=params.sharpen,
        )
        t0 = time.perf_counter()
        out = ltm.render_candidate(base_gray, cand)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        winning_algorithm = cand.algorithm
        winning_threshold = cand.threshold

    # Auto-mirror para back-engrave (PhotoGrav-style): si el material termina en
    # "_back_engrave", voltear el PNG horizontalmente. Al grabar en la cara
    # posterior del acrílico, hay que invertir para que se vea correcto desde el
    # frente. Si el usuario ya está aplicando MirrorX en LightBurn, debe desactivar
    # este flag para evitar doble-mirror.
    mirrored = False
    if (
        params.auto_mirror_back_engrave
        and profile is not None
        and profile.name.endswith("_back_engrave")
    ):
        out = np.fliplr(out).copy()
        mirrored = True

    meta = {
        "output_size": [int(out.shape[1]), int(out.shape[0])],
        "ranking_short_px": ranking_short,
        "resolved_sharpen_radius_px": resolved_radius,
        "material": profile.name if profile else "",
        "spot_mm": profile.spot_mm if profile else None,
        "white_ratio": float((out == 255).mean()),
        "render_ms": elapsed_ms,
        "quality_mode": quality_mode,
        "winning_algorithm": winning_algorithm,
        "winning_threshold": winning_threshold,
        "auto_mirrored": mirrored,
    }
    if refine_debug:
        meta["refine"] = refine_debug
    return out, meta


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Laser Image Prep API",
    description="Backend del wizard SvelteKit para preparacion de imagenes laser-ready.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite dev
        "http://localhost:4173", "http://127.0.0.1:4173",  # Vite preview
        "http://localhost:18765", "http://127.0.0.1:18765",  # Static mount (mismo origen pero por las dudas)
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    # Cliente lee X-Process-Time-Ms, X-Output-Width, etc.; con allow_headers=["*"] solo
    # se permite request — para que JS los lea hay que expose_headers explícito.
    allow_headers=["*"],
    expose_headers=[
        "X-Process-Time-Ms", "X-Output-Width", "X-Output-Height", "X-White-Ratio",
        "X-Sharpen-Radius-Px", "X-Material", "X-Preset-Applied", "X-Algorithm",
        "X-Quality-Mode", "X-Refine-Candidates", "X-Refine-Best-Score",
        "X-Refine-Improvement", "X-Refine-Seconds", "X-Preset-Reason",
        "X-Sim-Sigma-Px", "X-Sim-Spot-Mm", "X-Sim-Dpi",
        "Content-Disposition",
    ],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        pass
    return HealthResponse(
        status="ok",
        model_loaded=laser_scoring._lpips_loss_fn is not None,
        cuda_available=cuda,
        repo_root=str(_REPO_ROOT),
    )


def _material_info_from_profile(p, source: str) -> MaterialInfo:
    return MaterialInfo(
        name=p.name, spot_mm=p.spot_mm, default_dpi=p.default_dpi,
        tone_response=p.tone_response, power_pct_range=p.power_pct_range,
        notes=p.notes, source=source,
        speed_mm_s_range=getattr(p, "speed_mm_s_range", (200.0, 600.0)),
        pass_through=getattr(p, "pass_through", True),
        mirror_x_required=getattr(p, "mirror_x_required", False),
        lightburn_invert=getattr(p, "lightburn_invert", False),
        focus_mm=getattr(p, "focus_mm", 0.0),
        machine_compat=getattr(p, "machine_compat", ""),
    )


@app.get("/api/materials", response_model=list[MaterialInfo])
def list_materials() -> list[MaterialInfo]:
    """Lista builtins + presets custom de presets/materials/."""
    results: list[MaterialInfo] = []
    builtins = [
        ("acrylic_back_engrave", "builtin"),
        ("acrylic_funsun_9060_back_engrave", "builtin"),
        ("wood_generic", "builtin"),
    ]
    for name, source in builtins:
        try:
            p = laser_physics.load_material_profile(name)
            results.append(_material_info_from_profile(p, source))
        except Exception:
            continue
    presets_dir = _REPO_ROOT / "presets" / "materials"
    if presets_dir.is_dir():
        for json_file in presets_dir.glob("*.json"):
            name = json_file.stem
            if any(r.name == name for r in results):
                continue
            try:
                p = laser_physics.load_material_profile(name, presets_dir=presets_dir)
                results.append(_material_info_from_profile(p, "custom"))
            except Exception:
                continue
    return results


@app.get("/api/recommended_settings/{material_name}", response_model=RecommendedSettings)
def get_recommended_settings(material_name: str) -> RecommendedSettings:
    """
    Devuelve la configuracion recomendada para LightBurn / CAM del laser, dado un material.

    Incluye: DPI, interval mm, power %, speed mm/s, pass-through, mirror, invert, focus mm.
    """
    presets_dir = _REPO_ROOT / "presets" / "materials"
    try:
        p = laser_physics.load_material_profile(material_name, presets_dir=presets_dir)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=f"material '{material_name}' no encontrado: {exc}") from exc
    d = p.recommended_settings_dict()
    return RecommendedSettings(**d)


@app.get("/api/presets")
def list_presets() -> list[dict]:
    """Catalogo de presets curados para que el wizard los muestre."""
    return laser_presets.list_presets_dict()


@app.post("/api/recommend_preset")
async def recommend_preset_endpoint(image: UploadFile = File(...)) -> dict:
    """Analiza la imagen y devuelve el preset recomendado + estadisticos + motivo."""
    img_bytes = await image.read()
    img = _load_image_from_bytes(img_bytes)
    rgb_arr = np.array(img.convert("RGB"))
    rec = laser_presets.recommend_preset(rgb_arr)
    return {
        "preset_name": rec.preset_name,
        "preset_label": rec.preset_label,
        "reason": rec.reason,
        "stats": {
            "mean": rec.stats.mean,
            "std": rec.stats.std,
            "extreme_ratio": rec.stats.extreme_ratio,
            "edge_density": rec.stats.edge_density,
        },
    }


@app.get("/api/algorithms", response_model=list[AlgorithmGroup])
def list_algorithms() -> list[AlgorithmGroup]:
    """Devuelve algoritmos agrupados por familia para que el wizard pueda mostrar select coherente."""
    diffusion = set(ltm.DIFFUSION_ALGORITHMS.keys())
    burkes_blue = set(ltm.BURKES_BLUE_VARIANTS.keys())
    named = set(ltm.NAMED_RENDERERS.keys())
    ordered = {"threshold", "bayer4", "bayer8", "blue_noise16", "blue_noise_vac32"}
    mixes = named - ordered
    return [
        AlgorithmGroup(family="ordered_dither", algorithms=sorted(ordered & named)),
        AlgorithmGroup(family="error_diffusion", algorithms=sorted(diffusion)),
        AlgorithmGroup(family="burkes_blue_variants", algorithms=sorted(burkes_blue)),
        AlgorithmGroup(family="mix_multipass", algorithms=sorted(mixes)),
    ]


def _build_result_headers(
    meta: dict, applied_preset: str, detector_reason: str, fallback_algorithm: str
) -> dict[str, str]:
    """Construye los headers HTTP que el cliente lee tras un /process exitoso."""
    headers = {
        "X-Process-Time-Ms": f"{meta['render_ms']:.1f}",
        "X-Output-Width": str(meta["output_size"][0]),
        "X-Output-Height": str(meta["output_size"][1]),
        "X-White-Ratio": f"{meta['white_ratio']:.4f}",
        "X-Sharpen-Radius-Px": f"{meta['resolved_sharpen_radius_px']:.3f}",
        "X-Material": meta.get("material") or "",
        "X-Preset-Applied": applied_preset or "",
        "X-Algorithm": meta.get("winning_algorithm") or fallback_algorithm,
        "X-Quality-Mode": meta.get("quality_mode", "fast"),
        "Content-Disposition": f'attachment; filename="laser_ready_{meta.get("winning_algorithm") or fallback_algorithm}.png"',
    }
    if meta.get("refine"):
        r = meta["refine"]
        headers["X-Refine-Candidates"] = str(r["candidates_evaluated"])
        headers["X-Refine-Best-Score"] = f"{r['best_score_v5']:.4f}"
        headers["X-Refine-Improvement"] = f"{r['improvement_v5']:.4f}"
        headers["X-Refine-Seconds"] = f"{r['refine_seconds']:.2f}"
    if detector_reason:
        # HTTP headers ASCII-only; eliminar acentos del motivo en español sin perder el sentido.
        headers["X-Preset-Reason"] = detector_reason.encode("ascii", "ignore").decode("ascii")
    return headers


def _encode_png(out: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(out, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _process_endpoint_body(image_bytes: bytes, params: ProcessParams) -> Response:
    img = _load_image_from_bytes(image_bytes)
    # Resolver preset (incluye auto-deteccion si preset=='auto')
    resolved_params, applied_preset, detector_reason = _resolve_preset_overrides(params, img)
    out, meta = _process_image(img, resolved_params)
    png = _encode_png(out)
    headers = _build_result_headers(meta, applied_preset, detector_reason, resolved_params.algorithm)
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/api/process")
async def process(
    image: UploadFile = File(...),
    params_json: str = Form("{}"),
) -> Response:
    """
    Procesa imagen con params JSON. Devuelve PNG 1-bit.

    Cuerpo: `multipart/form-data` con campo `image` (file) y `params_json` (string JSON).
    Default quality_mode='best' (HQ refinement con sobol + v5 scoring).
    """
    import json
    try:
        raw = json.loads(params_json)
        # /api/process default a HQ best (a menos que el cliente especifique fast)
        if "quality_mode" not in raw:
            raw["quality_mode"] = "best"
        params = ProcessParams(**raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"params_json invalido: {exc}") from exc
    return _process_endpoint_body(await image.read(), params)


@app.post("/api/preview")
async def preview(
    image: UploadFile = File(...),
    params_json: str = Form("{}"),
) -> Response:
    """Igual a /api/process pero forzando max_side=400 + quality_mode=fast para rapidez."""
    import json
    try:
        raw = json.loads(params_json)
        raw["max_side"] = 400  # forzar preview rapido
        raw["quality_mode"] = "fast"  # nunca HQ search en preview
        params = ProcessParams(**raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"params_json invalido: {exc}") from exc
    return _process_endpoint_body(await image.read(), params)


@app.post("/api/simulate")
async def simulate(
    image: UploadFile = File(...),
    material: str = Form(""),
    output_dpi: int = Form(169),
    appearance: str = Form(""),
    background: int = Form(-1),
) -> Response:
    """
    Simula la apariencia del grabado fisico a partir de un PNG 1-bit.

    - `image`: PNG binario (255 = laser ON).
    - `material`: nombre del MaterialProfile (opcional; si vacio usar appearance).
    - `output_dpi`: DPI del grabado.
    - `appearance`: "acrylic_frost" | "wood_burn" | "raw" si no se pasa material.
    - `background`: 0..255 override del fondo (-1 = default).

    Devuelve PNG (mode L) representando como se veria grabado fotografiado.
    """
    img_bytes = await image.read()
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"imagen invalida: {exc}") from exc
    binary = np.array(pil, dtype=np.uint8)
    if not set(np.unique(binary).tolist()).issubset({0, 255}):
        # Auto-threshold si no es estrictamente binario
        binary = np.where(binary >= 128, 255, 0).astype(np.uint8)

    bg_value = background if background >= 0 else None
    if material:
        try:
            profile = laser_physics.load_material_profile(
                material, presets_dir=_REPO_ROOT / "presets" / "materials"
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=f"material '{material}': {exc}") from exc
        sim = laser_simulator.simulate_from_material_profile(binary, profile, output_dpi=output_dpi)
        material_label = profile.name
        spot_mm_used = profile.spot_mm
    else:
        if appearance not in ("acrylic_frost", "wood_burn", "raw"):
            raise HTTPException(
                status_code=400,
                detail=f"appearance '{appearance}' invalido (use acrylic_frost|wood_burn|raw)",
            )
        # Sin material, asumimos spot 0.15 mm como default razonable
        sim = laser_simulator.simulate_engraving(
            binary, spot_mm=0.15, output_dpi=output_dpi,
            material_appearance=appearance, background_value=bg_value,
        )
        material_label = ""
        spot_mm_used = 0.15

    sigma = laser_simulator.compute_spot_sigma_px(spot_mm_used, output_dpi)
    buf = io.BytesIO()
    Image.fromarray(sim, mode="L").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    headers = {
        "X-Sim-Sigma-Px": f"{sigma:.3f}",
        "X-Sim-Spot-Mm": f"{spot_mm_used:.3f}",
        "X-Sim-Dpi": str(output_dpi),
        "X-Material": material_label,
        "Content-Disposition": "attachment; filename=\"engraving_simulation.png\"",
    }
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


# ---------------------------------------------------------------------------
# Procesamiento asíncrono con progreso por SSE
# ---------------------------------------------------------------------------


# Singleton psutil.Process del worker (necesario para que cpu_percent() mantenga su
# baseline interno entre llamadas; un Process nuevo cada call devuelve 0.0).
# v2.1: thread-safe con lock para evitar corrupción si 2 workers concurrentes leen
# /escriben el estado interno de psutil simultáneamente (race documented por audit).
import threading as _threading

_PSUTIL_PROC: Any = None
_PSUTIL_LOCK = _threading.Lock()


def _init_process_metrics() -> None:
    """Inicializa el objeto psutil.Process global y dispara el baseline de cpu_percent."""
    global _PSUTIL_PROC
    with _PSUTIL_LOCK:
        try:
            import psutil
            if _PSUTIL_PROC is None:
                _PSUTIL_PROC = psutil.Process()
            # Primer call siempre devuelve 0; el siguiente medirá el % real
            _PSUTIL_PROC.cpu_percent(interval=None)
        except Exception:
            _PSUTIL_PROC = None


def _process_metrics() -> tuple[float | None, float | None]:
    """
    Devuelve (memoria_MB_RSS, cpu_pct) del proceso actual via psutil.

    Si psutil no está instalado, devuelve (None, None) sin romper.
    El cpu_pct es no-bloqueante (cached desde el último call al mismo objeto Process);
    `_init_process_metrics()` debe llamarse antes para crear el singleton.

    Thread-safe: protegido con _PSUTIL_LOCK contra acceso concurrente.
    """
    with _PSUTIL_LOCK:
        p = _PSUTIL_PROC
        if p is None:
            return None, None
        try:
            mem_mb = p.memory_info().rss / (1024 * 1024)
            cpu = p.cpu_percent(interval=None)
            return float(mem_mb), float(cpu)
        except Exception:
            return None, None


def _run_job_in_thread(job_id: str, image_bytes: bytes, params: ProcessParams) -> None:
    """
    Worker (thread) que procesa la imagen y reporta progreso + telemetría al JobRegistry.

    Se invoca desde el endpoint async via `asyncio.to_thread`. Toda la lógica
    pesada (decode, preprocess, render, HQ refine) corre acá.
    El JobState se manipula directamente (single-writer) y los readers (SSE/status)
    leen consistentemente porque el registry mantiene la referencia bajo lock.
    """
    job = REGISTRY.get(job_id)
    if job is None:
        return  # creado y borrado entre medio; nada que hacer.

    # Inicializar baseline del cpu_percent (sin esto la primera medición real es 0.0)
    _init_process_metrics()

    t_started = time.perf_counter()

    def _set_stage(new_stage: str, msg: str | None = None) -> None:
        job.stage = new_stage
        if msg:
            job.log(msg)
        else:
            job.log(f"stage → {new_stage}")
        job.elapsed_seconds = time.perf_counter() - t_started
        mem, cpu = _process_metrics()
        if mem is not None:
            job.memory_mb = mem
        if cpu is not None:
            job.cpu_pct = cpu
        job.touch()

    try:
        job.status = "running"
        job.log(f"job {job_id} iniciado (quality_mode={params.quality_mode})")
        _set_stage("decode", "decodificando imagen…")

        img = _load_image_from_bytes(image_bytes)
        job.log(f"imagen {img.size[0]}×{img.size[1]} OK")
        _set_stage("resolve_preset", "resolviendo preset…")
        resolved_params, applied_preset, detector_reason = _resolve_preset_overrides(params, img)
        if applied_preset:
            job.log(f"preset aplicado: {applied_preset}" + (f" ({detector_reason})" if detector_reason else ""))
        if resolved_params.material:
            job.log(f"material: {resolved_params.material}")

        # Setear total estimado (1 base + 24 sobol). El HQ refine confirma el total via cb.
        job.total = 25
        _set_stage("processing", f"procesando ({resolved_params.quality_mode})…")

        # Estado para el callback: tiempo del último update para calcular delta
        last_t = {"v": time.perf_counter(), "current": 0, "best": float("inf")}

        def progress_cb(current: int, total: int, best_score: float, elapsed: float) -> None:
            now = time.perf_counter()
            delta = now - last_t["v"]
            last_t["v"] = now
            last_t["current"] = current

            job.current = int(current)
            job.total = int(total)
            job.best_score = float(best_score)
            job.elapsed_seconds = float(elapsed)
            job.eta_seconds = (elapsed / max(1, current)) * max(0, total - current)
            job.seconds_per_candidate = elapsed / max(1, current)
            job.last_candidate_seconds = delta
            job.stage = "refine"
            job.push_score(best_score)

            # Si el best_score mejoró, loguear el evento
            if best_score < last_t["best"] - 1e-6:
                job.log(f"#{current}/{total}: nuevo mejor score {best_score:.4f} (Δ={delta:.2f}s)")
                last_t["best"] = best_score

            mem, cpu = _process_metrics()
            if mem is not None:
                job.memory_mb = mem
            if cpu is not None:
                job.cpu_pct = cpu
            job.touch()

        def cancel_check() -> bool:
            return bool(job.is_cancel_requested())

        out, meta = _process_image(img, resolved_params, progress_cb=progress_cb, cancel_check=cancel_check)

        if job.is_cancel_requested():
            job.status = "cancelled"
            job.stage = "cancelled"
            job.elapsed_seconds = time.perf_counter() - t_started
            job.log("cancelado por el cliente", kind="warn")
            job.touch()
            return

        _set_stage("encode", "comprimiendo PNG final…")
        png = _encode_png(out)
        headers = _build_result_headers(meta, applied_preset, detector_reason, resolved_params.algorithm)
        job.log(f"PNG {len(png) // 1024} KB, {meta['output_size'][0]}×{meta['output_size'][1]}")

        job.image_bytes = png
        job.image_headers = headers
        if meta.get("refine"):
            r = meta["refine"]
            job.current = int(r.get("candidates_evaluated", 1))
            job.total = int(r.get("candidates_total", 1))
        else:
            job.current = job.total = 1
        job.elapsed_seconds = time.perf_counter() - t_started
        job.eta_seconds = 0.0
        job.stage = "done"
        job.status = "done"
        job.log(f"finalizado en {job.elapsed_seconds:.2f}s ✓")
        mem, cpu = _process_metrics()
        if mem is not None:
            job.memory_mb = mem
        job.touch()
    except HTTPException as exc:
        job.status = "error"
        job.stage = "error"
        job.error_message = f"{exc.status_code}: {exc.detail}"
        job.elapsed_seconds = time.perf_counter() - t_started
        job.log(f"HTTPException {exc.status_code}: {exc.detail}", kind="error")
        job.touch()
    except Exception as exc:  # pragma: no cover - red de seguridad
        job.status = "error"
        job.stage = "error"
        job.error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        job.elapsed_seconds = time.perf_counter() - t_started
        job.log(f"error: {type(exc).__name__}: {exc}", kind="error")
        job.touch()


@app.post("/api/process_async")
async def process_async(
    image: UploadFile = File(...),
    params_json: str = Form("{}"),
) -> dict:
    """
    Procesa imagen en background y devuelve un `job_id` para seguir el progreso por SSE.

    Cliente: POST aquí → recibís `{"job_id": "..."}`. Después abrir EventSource a
    `/api/jobs/{job_id}/stream` para progreso y GET `/api/jobs/{job_id}/result` para el PNG.
    """
    try:
        raw = json.loads(params_json)
        if "quality_mode" not in raw:
            raw["quality_mode"] = "best"
        params = ProcessParams(**raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"params_json invalido: {exc}") from exc

    image_bytes = await image.read()
    # Validación temprana — si la imagen es inválida fallamos acá sin crear job.
    try:
        _load_image_from_bytes(image_bytes)
    except HTTPException:
        raise

    job = REGISTRY.create(total=25)
    # Disparamos el worker en thread separado (CPU-bound). Usamos to_thread para no
    # bloquear el event loop. asyncio.create_task NO ejecuta — programa la coroutine.
    asyncio.create_task(asyncio.to_thread(_run_job_in_thread, job.job_id, image_bytes, params))
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Snapshot puntual del estado de un job (sin streaming). Útil para polling simple."""
    job = REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' no existe o expiró")
    return job.to_progress_dict()


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> StreamingResponse:
    """
    Stream Server-Sent Events con el progreso del job hasta que llegue a done/error/cancelled.

    Cada evento es `data: <json>\\n\\n` con el snapshot to_progress_dict().
    Frecuencia ≈ 2 Hz (poll cada 500 ms). Termina con un evento final del estado terminal
    o un evento 'gone' si el job desaparece (limpieza por TTL).
    """
    job = REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' no existe o expiró")

    async def event_generator():
        last_payload = ""
        # Heartbeat inicial para que el cliente reciba algo de inmediato
        try:
            yield f"data: {json.dumps(job.to_progress_dict())}\n\n"
        except Exception:
            pass

        terminal = {"done", "error", "cancelled"}
        for _ in range(60 * 60 * 2):  # tope ~2h por si algo se cuelga: 0.5s * 14400 = 2h
            j = REGISTRY.get(job_id)
            if j is None:
                yield 'event: gone\ndata: {"status": "gone"}\n\n'
                return
            payload = json.dumps(j.to_progress_dict())
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if j.status in terminal:
                return
            await asyncio.sleep(0.5)

    headers = {
        # Headers críticos para SSE detrás de proxies y para que el browser no buffer-ee
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    """
    Devuelve el PNG final de un job completado. 404 si no existe, 409 si todavía no terminó.
    """
    job = REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' no existe o expiró")
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error_message or "error desconocido")
    if job.status == "cancelled":
        raise HTTPException(status_code=410, detail="job cancelado por el cliente")
    if job.status != "done" or job.image_bytes is None:
        raise HTTPException(status_code=409, detail=f"job aún no terminó (status={job.status})")
    return Response(content=job.image_bytes, media_type="image/png", headers=job.image_headers or {})


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    """Marca el job como 'cancel requested'. El worker abortará en el próximo checkpoint."""
    job = REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' no existe o expiró")
    job.request_cancel()
    return {"job_id": job_id, "cancel_requested": True, "status": job.status}


# ---------------------------------------------------------------------------
# Static frontend mount (sirve el build de SvelteKit en /app)
# ---------------------------------------------------------------------------
try:
    from fastapi.staticfiles import StaticFiles
    _web_build_dir = _REPO_ROOT / "web" / "build"
    if _web_build_dir.is_dir():
        app.mount("/app", StaticFiles(directory=str(_web_build_dir), html=True), name="webapp")
except ImportError:
    pass


@app.get("/")
def root_redirect() -> Response:
    """Redirige raiz a /app si esta el build, sino mensaje informativo."""
    web_build = _REPO_ROOT / "web" / "build" / "index.html"
    if web_build.is_file():
        return Response(
            status_code=302,
            headers={"Location": "/app/"},
        )
    return JSONResponse({
        "status": "API up",
        "wizard": "no build static encontrado; correr 'cd web && npm run build' o usar 'npm run dev' en :5173",
        "api_docs": "/docs",
    })


# Entry point para uvicorn programatico (opcional)
def serve(host: str = "127.0.0.1", port: int = 18765, reload: bool = False) -> None:
    import uvicorn
    uvicorn.run("scripts.api_server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve(reload=True)
