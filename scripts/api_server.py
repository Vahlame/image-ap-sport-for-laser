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

import io
import sys
import time
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

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
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
    preprocess_mode: str = Field(default="sauvola", description="Pre-CV: none|sauvola|niblack|grabcut|chanvese|sam2")
    max_side: int = Field(default=0, ge=0, le=8000, description="0 = no resize (full-res). Preview pasa 400.")


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


class AlgorithmGroup(BaseModel):
    family: str
    algorithms: list[str]


# ---------------------------------------------------------------------------
# Helpers de procesamiento
# ---------------------------------------------------------------------------


def _load_image_from_bytes(data: bytes) -> Image.Image:
    """Decodifica bytes a PIL Image RGB (raise HTTPException si invalido)."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
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
    # grabcut/chanvese/sam2 requieren mas args; los exponemos como preprocess avanzado en CLI
    raise HTTPException(
        status_code=400,
        detail=f"preprocess_mode '{mode}' no soportado en API (usar CLI o uno de: none, sauvola, niblack)",
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


def _hq_refine(base_gray: np.ndarray, base_params: ProcessParams) -> tuple[np.ndarray, dict, ltm.Candidate]:
    """
    HQ refinement: corre un sobol-search local (~24 variantes alrededor de los params dados)
    y devuelve el mejor por score v5 (no-reference: HVS-MSE + spectral radial + tone post-LUT).

    Cuesta tiempo (~30-90s para imágenes full-res en CPU; menos con CUDA en LPIPS).

    Sobol explora 5 dimensiones:
        threshold ± 16
        contrast ± 0.20
        brightness ± 12
        gamma ± 0.20
        sharpen ± 30

    Returns: (best_binary_output, debug_dict, winning_candidate)
    """
    from scipy.stats import qmc

    # base candidate
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

    # Sobol 5D, 24 puntos (siempre incluimos el baseline como punto 0)
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

    # Score v5 sin referencia: usamos el base_gray como "gris ideal"
    target_gray_v5 = base_gray.astype(np.float64)
    dummy_binary = np.zeros_like(base_gray, dtype=np.uint8)
    dummy_density = np.zeros((max(1, base_gray.shape[0] // 4), max(1, base_gray.shape[1] // 4)), dtype=np.float64)
    dummy_edges = np.zeros_like(base_gray, dtype=np.float64)

    best_score = float("inf")
    best_out: np.ndarray | None = None
    best_cand: ltm.Candidate | None = None
    scores_log: list[tuple[float, ltm.Candidate]] = []
    t0 = time.perf_counter()
    for cand in variants:
        out = ltm.render_candidate(base_gray, cand)
        score, *_ = laser_scoring.score_candidate_dispatch(
            "v5", out, target_gray_v5, dummy_binary, dummy_density, dummy_edges, cand,
        )
        scores_log.append((float(score), cand))
        if score < best_score:
            best_score = score
            best_out = out
            best_cand = cand
    elapsed = time.perf_counter() - t0

    assert best_out is not None and best_cand is not None
    debug = {
        "candidates_evaluated": len(variants),
        "best_score_v5": best_score,
        "baseline_score_v5": scores_log[0][0],
        "improvement_v5": scores_log[0][0] - best_score,
        "refine_seconds": elapsed,
    }
    return best_out, debug, best_cand


def _process_image(img_rgb: Image.Image, params: ProcessParams) -> tuple[np.ndarray, dict]:
    """
    Pipeline completo: resize -> preprocess -> apply LUT material -> render candidate.
    Devuelve (binario uint8, metadata dict).
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
        out, refine_debug, winning_cand = _hq_refine(base_gray, params)
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
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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


@app.get("/api/materials", response_model=list[MaterialInfo])
def list_materials() -> list[MaterialInfo]:
    """Lista builtins + presets custom de presets/materials/."""
    results: list[MaterialInfo] = []
    # Builtins programaticos
    builtins = [
        ("acrylic_back_engrave", "builtin"),
        ("wood_generic", "builtin"),
    ]
    for name, source in builtins:
        try:
            p = laser_physics.load_material_profile(name)
            results.append(MaterialInfo(
                name=p.name, spot_mm=p.spot_mm, default_dpi=p.default_dpi,
                tone_response=p.tone_response, power_pct_range=p.power_pct_range,
                notes=p.notes, source=source,
            ))
        except Exception:
            continue
    # JSON custom (si existe presets/materials/)
    presets_dir = _REPO_ROOT / "presets" / "materials"
    if presets_dir.is_dir():
        for json_file in presets_dir.glob("*.json"):
            name = json_file.stem
            if any(r.name == name for r in results):
                continue
            try:
                p = laser_physics.load_material_profile(name, presets_dir=presets_dir)
                results.append(MaterialInfo(
                    name=p.name, spot_mm=p.spot_mm, default_dpi=p.default_dpi,
                    tone_response=p.tone_response, power_pct_range=p.power_pct_range,
                    notes=p.notes, source="custom",
                ))
            except Exception:
                continue
    return results


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


def _process_endpoint_body(image_bytes: bytes, params: ProcessParams) -> Response:
    img = _load_image_from_bytes(image_bytes)
    # Resolver preset (incluye auto-deteccion si preset=='auto')
    resolved_params, applied_preset, detector_reason = _resolve_preset_overrides(params, img)
    out, meta = _process_image(img, resolved_params)
    buf = io.BytesIO()
    Image.fromarray(out, mode="L").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    headers = {
        "X-Process-Time-Ms": f"{meta['render_ms']:.1f}",
        "X-Output-Width": str(meta["output_size"][0]),
        "X-Output-Height": str(meta["output_size"][1]),
        "X-White-Ratio": f"{meta['white_ratio']:.4f}",
        "X-Sharpen-Radius-Px": f"{meta['resolved_sharpen_radius_px']:.3f}",
        "X-Material": meta["material"],
        "X-Preset-Applied": applied_preset,
        "X-Algorithm": meta.get("winning_algorithm", resolved_params.algorithm),
        "X-Quality-Mode": meta.get("quality_mode", "fast"),
        "Content-Disposition": f"attachment; filename=\"laser_ready_{meta.get('winning_algorithm', resolved_params.algorithm)}.png\"",
    }
    if meta.get("refine"):
        r = meta["refine"]
        headers["X-Refine-Candidates"] = str(r["candidates_evaluated"])
        headers["X-Refine-Best-Score"] = f"{r['best_score_v5']:.4f}"
        headers["X-Refine-Improvement"] = f"{r['improvement_v5']:.4f}"
        headers["X-Refine-Seconds"] = f"{r['refine_seconds']:.2f}"
    if detector_reason:
        # Solo encabezados ASCII; trimmear acentos en valores via repr no es ideal,
        # pero el cliente igual lo muestra como info. Encode-safe:
        headers["X-Preset-Reason"] = detector_reason.encode("ascii", "ignore").decode("ascii")
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


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
