"""
Metricas de scoring v1..v5.

- v1: legacy densidad-dominada (target-based).
- v2: SSIM continuo + terminos legacy v1 + regularizacion (target-based; tiene meseta intra-dither).
- v3: v2 + SSIM(blur) salida-vs-target_binario (target-based; suaviza halftone perceptualmente).
- v4: blur simetrico salida+target_continuo + SSIM + MSE + LPIPS(Alex) + legacy (target-based; perceptual).
- v5: **sin referencia (no-reference)**. HVS-MSE (CSF Mannos-Sakrison) + spectral radial penalty
  (validar perfil blue-noise) + tone match local post-LUT. Disenado para evaluar calidad
  fisica del grabado sin depender de un PNG objetivo. Ver `MEMORY-laser-snips` §"Fisica CO2".
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

from laser_runtime_env import apply_cuda_process_memory_cap, lpips_device_mode

# Modelo LPIPS cargado una vez por proceso (pesado en multiproceso).
_lpips_loss_fn: Any | None = None

# Contador para throttle de torch.cuda.empty_cache() en _lpips_distance_scaled.
# Sin throttle el caché CUDA acumula VRAM y los sweeps largos mueren ~400-500s.
_lpips_call_count: int = 0


def candidate_regularization_terms(candidate: Any) -> dict[str, float]:
    """Penalizacion suave por contraste bajo, brillo extremo o sharpen alto (compartida v2/v3/v4)."""
    c = float(candidate.contrast)
    b = float(candidate.brightness)
    sh = float(candidate.sharpen)
    reg_contrast = 0.20 * max(0.0, 0.70 - c)
    reg_brightness = 0.15 * (max(0.0, abs(b) - 25.0) / 40.0)
    reg_sharpen = 0.10 * (max(0.0, sh - 100.0) / 100.0)
    reg = reg_contrast + reg_brightness + reg_sharpen
    return {
        "reg": reg,
        "reg_contrast": reg_contrast,
        "reg_brightness": reg_brightness,
        "reg_sharpen": reg_sharpen,
    }


def _lpips_torch_device():
    """Dispositivo LPIPS: env LASER_LPIPS_DEVICE=auto|cuda|cpu (default auto)."""
    import torch

    mode = lpips_device_mode()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LASER_LPIPS_DEVICE=cuda pero torch.cuda.is_available() es False. "
                "Instala PyTorch con soporte CUDA (ver README: GPU + Score v4) o usa LASER_LPIPS_DEVICE=cpu."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_lpips_eval():
    """Inicializa Alex LPIPS lazy (requiere torch + paquete lpips)."""
    global _lpips_loss_fn
    if _lpips_loss_fn is None:
        apply_cuda_process_memory_cap(quiet=True)  # sync con Cursor 2026-05-15 (VRAM cap)
        try:
            import torch
            import lpips as lpips_pkg
        except ImportError as exc:
            raise ImportError(
                "Score v4 requiere dependencias perceptuales: pip install -e \".[perceptual]\" "
                "(torch, torchvision, lpips)."
            ) from exc
        device = _lpips_torch_device()
        # torchvision.models lanza UserWarning deprecacion pretrained/weights al cargar Alex vía lpips.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")
            _lpips_loss_fn = lpips_pkg.LPIPS(net="alex").to(device).eval()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    return _lpips_loss_fn


def _resize_gray_01_pair_for_lpips(
    a01: np.ndarray, b01: np.ndarray, min_side: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Alex/LPIPS necesita area suficiente; upscale bilineal si el lado corto es muy pequeño."""
    h, w = a01.shape[:2]
    m = min(h, w)
    if m >= min_side:
        return a01.astype(np.float64), b01.astype(np.float64)
    scale = float(min_side) / float(max(1, m))
    nw = max(min_side, int(round(w * scale)))
    nh = max(min_side, int(round(h * scale)))
    ia = (np.clip(a01, 0.0, 1.0) * 255.0).astype(np.uint8)
    ib = (np.clip(b01, 0.0, 1.0) * 255.0).astype(np.uint8)
    ra = (
        np.array(Image.fromarray(ia, mode="L").resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    )
    rb = (
        np.array(Image.fromarray(ib, mode="L").resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    )
    return ra, rb


def _lpips_distance_scaled(ob01: np.ndarray, tb01: np.ndarray) -> float:
    """
    Distancia LPIPS entre dos campos HxW en [0,1] (p.ej. tras blur gaussiano).
    Replica canal L→RGB y escala a [-1,1]. Devuelve termino en [0,1] (clamp suave).
    Si la imagen es muy pequeña para AlexNet, reescala bilinealmente (min lado 64 px).
    """
    ob01, tb01 = _resize_gray_01_pair_for_lpips(ob01, tb01)
    import torch

    loss_fn = _get_lpips_eval()
    device = next(loss_fn.parameters()).device
    with torch.no_grad():
        s1 = np.stack([ob01, ob01, ob01], axis=0).astype(np.float32)
        s2 = np.stack([tb01, tb01, tb01], axis=0).astype(np.float32)
        x1 = torch.from_numpy(s1).unsqueeze(0).to(device) * 2.0 - 1.0
        x2 = torch.from_numpy(s2).unsqueeze(0).to(device) * 2.0 - 1.0
        d = loss_fn(x1, x2)
        raw = float(d.mean().detach().cpu())
        # Liberar referencias intermedias antes de empty_cache.
        del s1, s2, x1, x2, d
    # Liberar el caché CUDA cada N llamadas para prevenir muerte ~400-500s por VRAM acumulada.
    # Sin esto, sweeps largos (>70 candidatos full-res v4) se cuelgan/mueren silenciosamente en
    # hosts con VRAM cap de ~5.5 GiB. Costo: ~1ms cada 25 evaluaciones.
    if device.type == "cuda":
        global _lpips_call_count
        _lpips_call_count += 1
        if _lpips_call_count % 25 == 0:
            torch.cuda.empty_cache()
    clipped = min(max(raw, 0.0), 2.0) / 2.0
    return clipped


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


def score_candidate_v2_terms(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any,
) -> dict[str, float]:
    """
    Desglose completo de score v2 (misma formula que score_candidate_v2).

    Terminos crudos:
      - ssim_raw: SSIM 0..1 entre salida y target en luminancia continua normalizada.
      - ssim_term: 1 - ssim_raw (cuanto peor vs SSIM perfecto 1).
      - pixel_error: MSE salida vs target **binarizado** (0..1).
      - edge_error / density_error: como en v1 pero sobre la salida vs mapas del target.
      - ratio_error: |fraccion blancos salida - fraccion blancos target|.
      - reg: penalizacion suave por contraste bajo, |brillo| extremo o sharpen alto.

    Pesos del score final:
      0.40*ssim_term + 0.20*pixel + 0.15*edge + 0.10*density + 0.05*ratio + 0.10*reg
    """
    out_norm = out.astype(np.float64) / 255.0
    tgt_cont = np.clip(target_gray.astype(np.float64) / 255.0, 0.0, 1.0)
    ssim_raw = float(structural_similarity(out_norm, tgt_cont, data_range=1.0))
    ssim_term = 1.0 - ssim_raw

    target_norm_bin = target_binary.astype(np.float64) / 255.0
    pixel_error = float(np.mean((out_norm - target_norm_bin) ** 2))
    density_error = float(np.mean((density_map(out) - target_density) ** 2))
    edge_error = float(np.mean(np.abs(edge_map(out) - target_edges)))
    white_ratio = float(np.mean(out == 255))
    target_white_ratio = float(np.mean(target_binary == 255))
    ratio_error = abs(white_ratio - target_white_ratio)

    rg = candidate_regularization_terms(candidate)
    reg = float(rg["reg"])
    reg_contrast = float(rg["reg_contrast"])
    reg_brightness = float(rg["reg_brightness"])
    reg_sharpen = float(rg["reg_sharpen"])

    w_ssim = 0.40 * ssim_term
    w_pixel = 0.20 * pixel_error
    w_edge = 0.15 * edge_error
    w_density = 0.10 * density_error
    w_ratio = 0.05 * ratio_error
    w_reg = 0.10 * reg
    score = w_ssim + w_pixel + w_edge + w_density + w_ratio + w_reg

    return {
        "ssim_raw": ssim_raw,
        "ssim_term": ssim_term,
        "pixel_error": pixel_error,
        "edge_error": edge_error,
        "density_error": density_error,
        "ratio_error": ratio_error,
        "reg": reg,
        "reg_contrast": reg_contrast,
        "reg_brightness": reg_brightness,
        "reg_sharpen": reg_sharpen,
        "white_ratio": white_ratio,
        "target_white_ratio": target_white_ratio,
        "w_ssim": w_ssim,
        "w_pixel": w_pixel,
        "w_edge": w_edge,
        "w_density": w_density,
        "w_ratio": w_ratio,
        "w_reg": w_reg,
        "score": score,
    }


def score_candidate_v3_terms(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any,
    blur_sigma: float = 1.15,
) -> dict[str, float]:
    """
    Score v3: maximizar similitud global reforzando alineacion estructural suavizada (halftone vs mascara).

    Reutiliza los terminos de v2 y anade **SSIM entre salida y target binario** tras un suavizado gaussiano
    (sigma ~1.15 px), util cuando el target es binario y la salida es 1-bit: el SSIM crudo en 1-bit es debil;
    el blur expone densidad local comparable a la vista humana a media distancia.

    Pesos (suman 1.0):
      0.28*ssim_cont + 0.22*ssim_blur_bin + 0.18*pixel + 0.14*edge + 0.09*density + 0.04*ratio + 0.05*reg
    """
    t = score_candidate_v2_terms(
        out, target_gray, target_binary, target_density, target_edges, candidate
    )
    out_norm = out.astype(np.float64) / 255.0
    tgt_bin = target_binary.astype(np.float64) / 255.0
    sig = float(blur_sigma)
    ob = gaussian_filter(out_norm, sigma=sig, mode="nearest")
    bb = gaussian_filter(tgt_bin, sigma=sig, mode="nearest")
    ssim_blur_raw = float(structural_similarity(ob, bb, data_range=1.0))
    ssim_blur_term = 1.0 - ssim_blur_raw

    w_ssim = 0.28 * float(t["ssim_term"])
    w_blur = 0.22 * ssim_blur_term
    w_pixel = 0.18 * float(t["pixel_error"])
    w_edge = 0.14 * float(t["edge_error"])
    w_density = 0.09 * float(t["density_error"])
    w_ratio = 0.04 * float(t["ratio_error"])
    w_reg = 0.05 * float(t["reg"])
    score = w_ssim + w_blur + w_pixel + w_edge + w_density + w_ratio + w_reg

    out2 = dict(t)
    out2.update(
        {
            "score": score,
            "ssim_blur_raw": ssim_blur_raw,
            "ssim_blur_term": ssim_blur_term,
            "blur_sigma": sig,
            "w_ssim": w_ssim,
            "w_ssim_blur": w_blur,
            "w_pixel": w_pixel,
            "w_edge": w_edge,
            "w_density": w_density,
            "w_ratio": w_ratio,
            "w_reg": w_reg,
        }
    )
    return out2


def score_candidate_v3(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any,
) -> tuple[float, float, float, float]:
    """Metrica v3 (ver score_candidate_v3_terms). Misma firma de retorno que v1/v2 para SQLite."""
    d = score_candidate_v3_terms(out, target_gray, target_binary, target_density, target_edges, candidate)
    return float(d["score"]), float(d["pixel_error"]), float(d["edge_error"]), float(d["white_ratio"])


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
    terms = score_candidate_v2_terms(
        out, target_gray, target_binary, target_density, target_edges, candidate
    )
    return (
        float(terms["score"]),
        float(terms["pixel_error"]),
        float(terms["edge_error"]),
        float(terms["white_ratio"]),
    )


def score_candidate_v4_terms(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any,
    blur_sigma: float = 1.15,
) -> dict[str, float]:
    """
    Score v4: sin SSIM continuo directo salida_binaria vs target_gris.

    - SSIM y MSE sobre **Gaussian blur simetrico** (misma sigma) aplicado a salida y target continuo.
    - **LPIPS(Alex)** sobre los mismos campos blurreados (L triplicado a RGB, [-1,1]),
      distancia clamp normalizada a [0,1] antes de ponderar. Si el lado corto es <64 px,
      se reescala bilinealmente solo para la pasada LPIPS (AlexNet necesita tamano minimo).
    - Conserva alineacion con mascara/bordes/densidad v2 y misma regularizacion.

    Pesos (suman 1.0):
      0.24*(1-SSIM_blur_sym) + 0.12*MSE_blur + 0.20*LPIPS_scaled + 0.15*pixel_bin
      + 0.12*edge + 0.07*density + 0.05*ratio + 0.05*reg

    Requiere: pip install -e ".[perceptual]"
    """
    out_norm = out.astype(np.float64) / 255.0
    tgt_cont = np.clip(target_gray.astype(np.float64) / 255.0, 0.0, 1.0)
    sig = float(blur_sigma)
    ob = gaussian_filter(out_norm, sigma=sig, mode="nearest")
    tb = gaussian_filter(tgt_cont, sigma=sig, mode="nearest")
    ssim_sym_raw = float(structural_similarity(ob, tb, data_range=1.0))
    ssim_sym_term = 1.0 - ssim_sym_raw
    mse_blur = float(np.mean((ob - tb) ** 2))

    lpips_term = _lpips_distance_scaled(ob.astype(np.float64), tb.astype(np.float64))

    target_norm_bin = target_binary.astype(np.float64) / 255.0
    pixel_error = float(np.mean((out_norm - target_norm_bin) ** 2))
    density_error = float(np.mean((density_map(out) - target_density) ** 2))
    edge_error = float(np.mean(np.abs(edge_map(out) - target_edges)))
    white_ratio = float(np.mean(out == 255))
    target_white_ratio = float(np.mean(target_binary == 255))
    ratio_error = abs(white_ratio - target_white_ratio)

    rg = candidate_regularization_terms(candidate)
    reg = float(rg["reg"])

    w_ssim_sym = 0.24 * ssim_sym_term
    w_mse_blur = 0.12 * mse_blur
    w_lpips = 0.20 * lpips_term
    w_pixel = 0.15 * pixel_error
    w_edge = 0.12 * edge_error
    w_density = 0.07 * density_error
    w_ratio = 0.05 * ratio_error
    w_reg = 0.05 * reg
    score = w_ssim_sym + w_mse_blur + w_lpips + w_pixel + w_edge + w_density + w_ratio + w_reg

    return {
        "blur_sigma": sig,
        "ssim_sym_raw": ssim_sym_raw,
        "ssim_sym_term": ssim_sym_term,
        "mse_blur": mse_blur,
        "lpips_term": lpips_term,
        "pixel_error": pixel_error,
        "edge_error": edge_error,
        "density_error": density_error,
        "ratio_error": ratio_error,
        "reg": reg,
        "reg_contrast": float(rg["reg_contrast"]),
        "reg_brightness": float(rg["reg_brightness"]),
        "reg_sharpen": float(rg["reg_sharpen"]),
        "white_ratio": white_ratio,
        "target_white_ratio": target_white_ratio,
        "w_ssim_sym": w_ssim_sym,
        "w_mse_blur": w_mse_blur,
        "w_lpips": w_lpips,
        "w_pixel": w_pixel,
        "w_edge": w_edge,
        "w_density": w_density,
        "w_ratio": w_ratio,
        "w_reg": w_reg,
        "score": score,
    }


def score_candidate_v4(
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any,
) -> tuple[float, float, float, float]:
    """Metrica v4 (ver score_candidate_v4_terms). Misma firma de retorno que v2/v3."""
    d = score_candidate_v4_terms(out, target_gray, target_binary, target_density, target_edges, candidate)
    return float(d["score"]), float(d["pixel_error"]), float(d["edge_error"]), float(d["white_ratio"])


# ---------------------------------------------------------------------------
# Score v5: sin referencia (no-reference quality)
# ---------------------------------------------------------------------------


def _csf_mannos_sakrison_filter(image: np.ndarray, ppd: float) -> np.ndarray:
    """
    Aplica filtro CSF (Mannos-Sakrison 1974) via FFT.

    CSF(f) = 2.6 * (0.0192 + 0.114*f) * exp(-(0.114*f)**1.1)  [cycles/degree]

    Modela respuesta del sistema visual humano: peak ~4 cpd, atenua baja y alta freq.
    Aplicado a un binario, produce la "imagen percibida" tras integrar el ojo a
    distancia normal de vision. Usado en v5 para HVS-MSE y para validar que la salida
    halftone reconstruye la luminancia del gris pretendido.
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    fy = (yy - cy) / float(h)
    fx = (xx - cx) / float(w)
    f_cpp = np.sqrt(fy * fy + fx * fx)
    f_cpd = np.maximum(f_cpp * float(ppd), 1e-6)
    csf = 2.6 * (0.0192 + 0.114 * f_cpd) * np.exp(-((0.114 * f_cpd) ** 1.1))
    csf = csf / csf.max()
    fft_shift = np.fft.fftshift(np.fft.fft2(img))
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * csf)))
    return filtered


def hvs_mse(binary: np.ndarray, gray_post_lut: np.ndarray, ppd: float = 64.0) -> float:
    """MSE entre binario y gris (post-LUT) tras pasar ambos por filtro CSF."""
    if binary.shape != gray_post_lut.shape:
        raise ValueError(f"[SCORING] hvs_mse: shapes distintas {binary.shape} vs {gray_post_lut.shape}")
    b_filt = _csf_mannos_sakrison_filter(binary, ppd)
    g_filt = _csf_mannos_sakrison_filter(gray_post_lut, ppd)
    return float(np.mean((b_filt - g_filt) ** 2))


def spectral_radial_penalty(binary: np.ndarray, low_band_fraction: float = 0.10) -> float:
    """
    Penaliza energia en bins radiales de baja frecuencia del power spectrum (DC excluido).

    Blue-noise ideal: energia uniformemente alta en alta freq, ~0 en baja.
    Clusters direccionales (Floyd sin serpentine, threshold puro en planos):
    energia alta en baja freq -> patrones visibles a distancia.

    Returns:
        ratio en [0, 1]: 0 = perfil blue-noise ideal; 1 = todo concentrado en baja frec.
    """
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
    if n_bins <= 1:
        return 0.0
    cutoff = max(1, int(round(n_bins * float(low_band_fraction))))
    low_energy = float(radial_avg[1:cutoff + 1].mean())
    total_energy = float(radial_avg[1:].mean()) + 1e-12
    return float(low_energy / (low_energy + total_energy))


def tone_match_error(binary: np.ndarray, gray_post_lut: np.ndarray, scale: int = 8) -> float:
    """
    Error tonal local: promedio del binario en bloques scale x scale debe matchear
    el gris post-LUT en la misma escala. Penaliza dot-gain no compensado.

    `gray_post_lut` puede ser el gris pre-dither (sin LUT = identidad) o el gris tras
    aplicar la LUT del material (compensa dot-gain del laser; ver `laser_physics`).
    """
    from scipy.ndimage import uniform_filter
    if binary.shape != gray_post_lut.shape:
        raise ValueError(f"[SCORING] tone_match: shapes distintas {binary.shape} vs {gray_post_lut.shape}")
    b_norm = binary.astype(np.float64) / 255.0
    g_norm = gray_post_lut.astype(np.float64)
    if g_norm.max() > 1.0:
        g_norm = g_norm / 255.0
    k = max(2, int(scale))
    b_local = uniform_filter(b_norm, size=k, mode="nearest")
    g_local = uniform_filter(g_norm, size=k, mode="nearest")
    return float(np.mean((b_local - g_local) ** 2))


def multi_scale_tone_match_error(
    binary: np.ndarray,
    gray_post_lut: np.ndarray,
    scales: tuple[int, ...] = (4, 8, 16),
    weights: tuple[float, ...] = (0.50, 0.30, 0.20),
) -> float:
    """
    Tone-match error promediado en MÚLTIPLES escalas.

    El single-scale (scale=8) original es ciego a detalles pequeños: un toro de 4-8 px
    en zona casi blanca se promedia con el blanco circundante y el error es bajo aunque
    el binario haya perdido completamente el detalle.

    Multi-scale: escalas finas (4 px) capturan detalles chicos; escalas gruesas (16 px)
    mantienen tono general. Peso mayor en escala fina (0.50) revela la pérdida del toro.

    Pesos por default suman 1.0 — el caller puede ajustarlos.
    """
    if len(scales) != len(weights):
        raise ValueError(f"[SCORING] scales y weights deben tener mismo largo: {len(scales)} vs {len(weights)}")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"[SCORING] weights de multi_scale_tone_match deben sumar 1.0, suman {sum(weights)}")
    total = 0.0
    for s, w in zip(scales, weights):
        total += w * tone_match_error(binary, gray_post_lut, scale=s)
    return float(total)


def edge_preservation_error(
    binary: np.ndarray,
    gray_post_lut: np.ndarray,
    ppd: float = 64.0,
    bright_threshold: float = 0.6,
    edge_percentile: float = 85.0,
) -> float:
    """
    Mide la pérdida de bordes finos en zonas brillantes (el problema del logo Red Bull
    en capó blanco).

    Pipeline:
      1. Sobel-magnitude del gris pre-dither → mapa de "dónde están los bordes importantes".
      2. Sobel-magnitude del binario PASADO por filtro CSF (HVS) → mapa de bordes percibidos.
      3. Máscara de zonas brillantes (gray > bright_threshold) → ahí es donde el problema duele.
      4. Bordes "importantes": top-K% del magnitude del gris (K = 100 - edge_percentile).
      5. Error = 1 - IoU entre bordes importantes del gris y del binario filtrado,
         ponderado por la intensidad del borde y restringido a zonas brillantes.

    Returns:
        score en [0, 1]: 0 = preserva todos los bordes en zonas claras; 1 = los pierde todos.

    Justificación física:
      Un cluster de puntos negros en zona casi blanca degrada el spectral_radial_penalty
      pero es **necesario** para que el ojo perciba el detalle. Este término compensa
      esa penalización injusta, premiando al algoritmo que sí preserva el toro.

    Args:
        bright_threshold: gris normalizado [0,1] mínimo para considerar "zona brillante"
            (default 0.6 = capó claro, paredes pastel, fondos blancos).
        edge_percentile: percentil sobre el magnitude para definir "edge importante"
            (default 85 = top 15% del gradient; ajustar si la imagen tiene mucho/poco edge).
    """
    from scipy import ndimage as ndi
    if binary.shape != gray_post_lut.shape:
        raise ValueError(
            f"[SCORING] edge_preservation: shapes distintas {binary.shape} vs {gray_post_lut.shape}"
        )

    g = gray_post_lut.astype(np.float64)
    if g.max() > 1.0:
        g = g / 255.0
    b = binary.astype(np.float64) / 255.0

    # Edges del gris (referencia) — Sobel magnitude
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    g_mag = np.sqrt(gx * gx + gy * gy)

    # Edges del binario PERCIBIDO (post-CSF) — esto es lo que el ojo ve realmente
    b_perceived = _csf_mannos_sakrison_filter(binary, ppd)
    bx = ndi.sobel(b_perceived, axis=1)
    by = ndi.sobel(b_perceived, axis=0)
    b_mag = np.sqrt(bx * bx + by * by)

    # Máscara de zonas brillantes — donde la pérdida importa
    bright_mask = g > float(bright_threshold)
    if not np.any(bright_mask):
        return 0.0  # no hay zonas brillantes → nada que preservar

    # Normalizar magnitudes ANTES de seleccionar bordes (Sobel max ~4.0 con kernel 3x3
    # sobre valor en [0,1]; normalizar a [0,1] facilita el threshold).
    g_max = float(g_mag.max())
    if g_max < 1e-6:
        return 0.0  # imagen plana sin bordes en ningún lado
    g_mag_norm = g_mag / g_max

    # Bordes "importantes": estrategia robusta a imágenes mayormente planas.
    # Usar el percentil sobre los magnitudes NO-cero (descartando el noise floor)
    # combinado con un mínimo absoluto del 5% del max. Así "important" significa
    # "borde real, no ruido de cuantización".
    nonzero_thresh = 0.05  # 5% del max — descarta ruido de Sobel en zonas planas
    nonzero_mag = g_mag_norm[g_mag_norm > nonzero_thresh]
    if nonzero_mag.size == 0:
        # Fallback: imagen "mayormente plana" pero puede tener un MICRO-DETALLE
        # aislado (logo Red Bull 4×6px, etc.) que no cruza el threshold por percentil
        # pero sí tiene gradient relativo distinguible.
        # Usamos un threshold absoluto muy bajo (1% del max) en zonas brillantes
        # como red de seguridad para no perder estos casos.
        micro_thresh = 0.01
        important_edges = (g_mag_norm >= micro_thresh) & bright_mask
        if not np.any(important_edges):
            return 0.0
    else:
        edge_thresh = float(np.percentile(nonzero_mag, float(edge_percentile)))
        edge_thresh = max(edge_thresh, nonzero_thresh)
        important_edges = (g_mag_norm >= edge_thresh) & bright_mask
        if not np.any(important_edges):
            return 0.0  # los bordes importantes están en zonas oscuras (no aplica)

    # Normalizar b_mag al mismo rango que g_mag para comparación 1:1
    b_max = float(b_mag.max())
    b_mag_norm = b_mag / b_max if b_max > 1e-12 else b_mag * 0.0

    # Error = diferencia normalizada en bordes importantes, ponderado por intensidad
    edge_weight = g_mag_norm[important_edges]
    diff = np.abs(g_mag_norm[important_edges] - b_mag_norm[important_edges])
    weighted_error = float(np.sum(diff * edge_weight) / (np.sum(edge_weight) + 1e-12))

    return float(min(max(weighted_error, 0.0), 1.0))


def score_candidate_v5_terms(
    out: np.ndarray,
    gray: np.ndarray,
    candidate: Any | None = None,
    *,
    lut: "Any | None" = None,
    ppd: float = 64.0,
    low_band_fraction: float = 0.10,
    tone_scale: int = 8,
    multi_scale_tone: bool = True,
    detail_weight: float = 0.35,
    bright_threshold: float = 0.6,
) -> dict[str, float]:
    """
    Desglose completo de score v5 sin referencia (v1.5+: con preservación de bordes).

    Combina:
      - HVS-MSE (CSF Mannos-Sakrison): error perceptual binario vs gris tras filtro.
      - Spectral radial penalty: energia en baja freq (blue-noise compliance).
      - Tone match local post-LUT (multi-scale si multi_scale_tone=True):
        match en bloques de [4, 8, 16] px ponderado, captura detalles finos.
      - **Edge preservation error** (v1.5): penaliza pérdida de bordes finos en
        zonas brillantes (problema del logo Red Bull en capó claro).
      - Regularizacion v2/v3/v4 compartida.

    Pesos (v1.5):
      0.40*hvs_mse + 0.25*spec_penalty + 0.15*tone + 0.15*detail + 0.05*reg

    Backwards-compat: si `multi_scale_tone=False`, usa solo scale=tone_scale (legacy v1.4)
    y omite el detail term (detail_weight=0.0 lo desactiva).

    `lut` callable: si se pasa, se aplica al gris antes de comparar (compensar dot-gain).
    `ppd` (pixels per degree): default 64 razonable para ~300 DPI viewing distance.
    `bright_threshold`: gris normalizado [0,1] sobre el cual se considera "zona brillante"
        para el edge preservation (default 0.6).
    """
    if out.shape != gray.shape:
        raise ValueError(f"[SCORING] v5 shapes distintas {out.shape} vs {gray.shape}")

    gray_lut = lut(gray) if lut is not None else gray
    hvs_err = hvs_mse(out, gray_lut, ppd=ppd)
    spec_penalty = spectral_radial_penalty(out, low_band_fraction=low_band_fraction)
    if multi_scale_tone:
        tone_err = multi_scale_tone_match_error(out, gray_lut)
    else:
        tone_err = tone_match_error(out, gray_lut, scale=tone_scale)

    # Detail preservation: solo si detail_weight > 0
    if detail_weight > 0:
        detail_err = edge_preservation_error(out, gray_lut, ppd=ppd, bright_threshold=bright_threshold)
    else:
        detail_err = 0.0

    white_ratio = float(np.mean(out == 255))

    if candidate is not None:
        rg = candidate_regularization_terms(candidate)
        reg = float(rg["reg"])
        reg_contrast = float(rg["reg_contrast"])
        reg_brightness = float(rg["reg_brightness"])
        reg_sharpen = float(rg["reg_sharpen"])
    else:
        reg = reg_contrast = reg_brightness = reg_sharpen = 0.0

    # Re-balance v1.5: 0.35*hvs + 0.20*spec + 0.10*tone + 0.35*detail + 0.05*reg = 1.05
    # Si detail_weight=0, redistribuir su peso a hvs/spec/tone (preserva semántica v1.4).
    if detail_weight > 0:
        # Pesos base ajustados; el caller puede subir detail_weight para casos extremos
        # (logos chicos en zonas planas), bajarlo si genera ruido (default OK para fotos
        # comunes con detalles visibles).
        w_hvs_factor = 0.35
        w_spec_factor = 0.20
        w_tone_factor = 0.10
        w_detail_factor = float(detail_weight)
    else:
        # Modo legacy v1.4 (sin detail term): preserva pesos originales 50/30/20.
        w_hvs_factor = 0.50
        w_spec_factor = 0.30
        w_tone_factor = 0.20
        w_detail_factor = 0.0

    w_hvs = w_hvs_factor * hvs_err
    w_spec = w_spec_factor * spec_penalty
    w_tone = w_tone_factor * tone_err
    w_detail = w_detail_factor * detail_err
    w_reg = 0.05 * reg
    score = w_hvs + w_spec + w_tone + w_detail + w_reg

    return {
        "hvs_mse": hvs_err,
        "spectral_lowfreq_penalty": spec_penalty,
        "tone_error": tone_err,
        "detail_error": detail_err,
        "white_ratio": white_ratio,
        "reg": reg,
        "reg_contrast": reg_contrast,
        "reg_brightness": reg_brightness,
        "reg_sharpen": reg_sharpen,
        "ppd": float(ppd),
        "low_band_fraction": float(low_band_fraction),
        "tone_scale": int(tone_scale),
        "multi_scale_tone": bool(multi_scale_tone),
        "detail_weight": float(detail_weight),
        "bright_threshold": float(bright_threshold),
        "w_hvs": w_hvs,
        "w_spec": w_spec,
        "w_tone": w_tone,
        "w_detail": w_detail,
        "w_reg": w_reg,
        "score": score,
    }


def score_candidate_v5(
    out: np.ndarray,
    gray: np.ndarray,
    candidate: Any | None = None,
    *,
    lut: "Any | None" = None,
    ppd: float = 64.0,
    low_band_fraction: float = 0.10,
    tone_scale: int = 8,
) -> tuple[float, float, float, float]:
    """
    Score v5 (ver score_candidate_v5_terms).

    Returns:
        (score, hvs_mse, spectral_lowfreq_penalty, white_ratio)

        Los slots 2-3 tienen semantica v5-especifica (no son `pixel_error`/`edge_error`
        como v1-v4). `white_ratio` se preserva en slot 4 por compatibilidad SQLite.
    """
    d = score_candidate_v5_terms(
        out, gray, candidate,
        lut=lut, ppd=ppd,
        low_band_fraction=low_band_fraction, tone_scale=tone_scale,
    )
    return (
        float(d["score"]),
        float(d["hvs_mse"]),
        float(d["spectral_lowfreq_penalty"]),
        float(d["white_ratio"]),
    )


# ---------------------------------------------------------------------------
# Dispatch unificado
# ---------------------------------------------------------------------------


ScoreVersion = Literal["v1", "v2", "v3", "v4", "v5"]


def score_candidate_dispatch(
    version: str,
    out: np.ndarray,
    target_gray: np.ndarray,
    target_binary: np.ndarray,
    target_density: np.ndarray,
    target_edges: np.ndarray,
    candidate: Any | None = None,
    *,
    lut: "Any | None" = None,
    ppd: float = 64.0,
) -> tuple[float, float, float, float]:
    """
    Enruta a v1..v5.

    - v1: legacy densidad-dominada (target-based).
    - v2..v4: target-based con SSIM/blur/LPIPS y regularizacion (requieren `candidate`).
    - v5: **sin referencia**. Usa `target_gray` como "gris ideal" (caller decide:
      gris del target externo, o gris pre-dither del propio input para no-reference puro).
      Acepta `lut` (callable gray->gray) y `ppd` por keyword.
    """
    if version == "v1":
        return score_candidate_v1(out, target_gray, target_binary, target_density, target_edges)
    if version == "v2":
        if candidate is None:
            raise ValueError("[SCORING] score-version v2 requiere candidate")
        return score_candidate_v2(out, target_gray, target_binary, target_density, target_edges, candidate)
    if version == "v3":
        if candidate is None:
            raise ValueError("[SCORING] score-version v3 requiere candidate")
        return score_candidate_v3(out, target_gray, target_binary, target_density, target_edges, candidate)
    if version == "v4":
        if candidate is None:
            raise ValueError("[SCORING] score-version v4 requiere candidate")
        return score_candidate_v4(out, target_gray, target_binary, target_density, target_edges, candidate)
    if version == "v5":
        return score_candidate_v5(out, target_gray, candidate, lut=lut, ppd=ppd)
    raise ValueError(f"[SCORING] version desconocida: {version!r}")
