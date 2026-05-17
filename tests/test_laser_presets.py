"""Tests del sistema de presets + auto-detector (`scripts/laser_presets.py`)."""

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


def test_catalog_has_all_expected_presets():
    lp = _load("laser_presets")
    names = {p.name for p in lp.ALL_PRESETS}
    assert "photo_general" in names
    assert "portrait" in names
    assert "scene_dark" in names
    assert "scene_bright" in names
    assert "poster_back_engrave" in names
    assert "line_art" in names
    # v1.7: nuevo preset para fotos con detalle fino sobre acrílico
    assert "photo_high_detail" in names


def test_photo_high_detail_has_conservative_params():
    """Preset photo_high_detail debe ser MÁS CONSERVADOR que photo_back_engrave."""
    lp = _load("laser_presets")
    hd = lp.get_preset("photo_high_detail")
    be = lp.get_preset("photo_back_engrave")
    # Gamma más cercano a lineal (preserva midtones)
    assert hd.gamma < be.gamma
    # Sharpen menor (no amplifica ruido)
    assert hd.sharpen < be.sharpen
    # Preprocess CLAHE para revelar detalle
    assert hd.preprocess_mode == "clahe"


def test_detector_chooses_high_detail_for_midtone_rich_image():
    """Imagen rica en midtones (extreme_ratio < 0.40) sobre acrílico debe ir a photo_high_detail."""
    lp = _load("laser_presets")
    # Gradiente suave: muchos midtones, pocos extremos
    h, w = 200, 200
    rng = np.random.default_rng(7)
    # Imagen mayormente gris medio con poca varianza extrema
    gray_lvl = rng.integers(80, 180, size=(h, w), dtype=np.uint8)
    rgb = np.stack([gray_lvl, gray_lvl, gray_lvl], axis=-1)
    rec = lp.recommend_preset(rgb, material="acrylic_funsun_9060_back_engrave")
    assert rec.preset_name == "photo_high_detail", (
        f"esperaba photo_high_detail, got {rec.preset_name}; extr={rec.stats.extreme_ratio*100:.0f}%"
    )


def test_detector_chooses_back_engrave_for_natural_bimodal_photo():
    """
    Foto NATURAL bimodal (Earth/rally car style): mucho contraste pero NO es dibujo.
    Característica: extreme_ratio alto (40-55%) PERO very_bright_ratio bajo (<18%).
    Debe elegir photo_back_engrave (no cartoon — cartoon requiere vbright > 50%).
    """
    lp = _load("laser_presets")
    rng = np.random.default_rng(42)
    h, w = 200, 200
    # Foto bimodal natural: zonas oscuras + zonas medio-claras con texturas (no blanco puro)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # 40% pixels oscuros (espacio/sombras), 40% pixels claros pero no puros (180-210)
    rgb[: int(h * 0.4)] = rng.integers(10, 35, size=(int(h * 0.4), w, 3), dtype=np.uint8)  # oscuro
    rgb[int(h * 0.4):] = rng.integers(180, 210, size=(h - int(h * 0.4), w, 3), dtype=np.uint8)  # claro NO puro
    rec = lp.recommend_preset(rgb, material="acrylic_funsun_9060_back_engrave")
    # extr ~80% (40 dark + 40 bright), vbright ~0% (claros son 180-210, no >215)
    # → cae en regla C default photo_back_engrave
    assert rec.preset_name == "photo_back_engrave", (
        f"esperaba photo_back_engrave para foto bimodal natural, got {rec.preset_name}; "
        f"stats: extr={rec.stats.extreme_ratio*100:.0f}% vbri={rec.stats.very_bright_ratio*100:.0f}%"
    )


def test_detector_handles_bright_background_with_midtone_subject():
    """
    Caso mujer kayak: fondo muy brillante (cielo + lago = >18% very_bright) +
    sujeto midtone (cabello, ropa). El preset agresivo aplastaría el sujeto a blanco.
    Debe elegir photo_high_detail.
    """
    lp = _load("laser_presets")
    h, w = 200, 200
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Fondo muy brillante (mucho > 215): 30% de la imagen
    rgb[:60] = 240  # cielo casi blanco
    # Zona oscura (árboles): 25%
    rgb[60:110, :] = 20  # árboles oscuros
    # Sujeto midtone (persona): 45% rango medio
    rgb[110:, :] = 130  # ropa/cabello midtone
    rec = lp.recommend_preset(rgb, material="acrylic_funsun_9060_back_engrave")
    # very_bright ratio será ~30%, extreme_ratio ~55% (30 bright + 25 dark)
    # → cae en la nueva regla B (very_bright > 18% AND extr < 75%)
    assert rec.preset_name == "photo_high_detail", (
        f"esperaba photo_high_detail (caso fondo brillante + sujeto midtone), "
        f"got {rec.preset_name}; stats: extr={rec.stats.extreme_ratio*100:.0f}% "
        f"vbright={rec.stats.very_bright_ratio*100:.0f}%"
    )


def test_image_stats_has_very_bright_ratio():
    """ImageStats incluye el nuevo campo very_bright_ratio."""
    lp = _load("laser_presets")
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb[:20] = 250  # 40% very bright
    s = lp.compute_image_stats(rgb)
    assert hasattr(s, "very_bright_ratio")
    assert 0.35 < s.very_bright_ratio < 0.45, f"esperaba ~0.40, got {s.very_bright_ratio}"


def test_cartoon_preset_in_catalog():
    """v1.8: el preset cartoon_back_engrave debe estar en el catálogo."""
    lp = _load("laser_presets")
    names = {p.name for p in lp.ALL_PRESETS}
    assert "cartoon_back_engrave" in names
    p = lp.get_preset("cartoon_back_engrave")
    # Threshold puro con umbral bajo + invert (convención correcta del proyecto)
    assert p.algorithm == "threshold"
    assert p.threshold <= 20, f"threshold debe ser bajo para detectar solo blanco puro, got {p.threshold}"
    assert p.invert is True
    assert p.sharpen == 0.0  # no sharpen extra
    assert p.preprocess_mode == "none"  # no preprocess sobre dibujos


def test_detector_chooses_cartoon_for_anime_with_white_bg():
    """
    Imagen tipo anime/illustration con fondo blanco grande (>50%) y bimodal
    (extr > 65%) debe elegir cartoon_back_engrave.
    """
    lp = _load("laser_presets")
    # Anime sintético: ~70% blanco puro + ~25% sujeto colorido oscuro
    h, w = 200, 200
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # fondo blanco completo
    rgb[60:140, 60:140] = (40, 180, 200)  # "sujeto" cyan medio en el centro
    rgb[80:120, 80:120] = (20, 20, 20)  # "detalles oscuros" interior

    rec = lp.recommend_preset(rgb, material="acrylic_funsun_9060_back_engrave")
    assert rec.preset_name == "cartoon_back_engrave", (
        f"esperaba cartoon_back_engrave, got {rec.preset_name}; "
        f"stats: extr={rec.stats.extreme_ratio*100:.0f}% vbri={rec.stats.very_bright_ratio*100:.0f}%"
    )


def test_fine_textures_preset_in_catalog():
    """v1.9: el preset photo_fine_textures debe estar en el catálogo con params correctos."""
    lp = _load("laser_presets")
    names = {p.name for p in lp.ALL_PRESETS}
    assert "photo_fine_textures" in names
    p = lp.get_preset("photo_fine_textures")
    assert p.algorithm == "stucki_serpentine"  # mejor que jarvis para texturas
    assert p.sharpen >= 100  # sharpen alto para realzar bordes finos
    assert p.preprocess_mode == "sauvola"  # contraste local sin amplificar ruido


def test_detector_chooses_fine_textures_for_high_edge_density(monkeypatch):
    """
    Imagen con texturas finas dominantes (edge_density > 10%, std > 50, extr < 50%)
    debe elegir photo_fine_textures.

    Validado empíricamente con Hokusai "Great Wave" (edge=10%, std=55, extr=9%).
    Acá usamos monkeypatch para inyectar stats sintéticos directamente — más confiable
    que fabricar una imagen con stats exactos.
    """
    lp = _load("laser_presets")
    fake_stats = lp.ImageStats(
        mean=150.0, std=60.0, extreme_ratio=0.10, edge_density=0.12, very_bright_ratio=0.05
    )
    monkeypatch.setattr(lp, "compute_image_stats", lambda rgb: fake_stats)
    rec = lp.recommend_preset(np.zeros((10, 10, 3), dtype=np.uint8), material="acrylic_funsun_9060")
    assert rec.preset_name == "photo_fine_textures", (
        f"esperaba photo_fine_textures, got {rec.preset_name}"
    )


def test_detector_size_invariant():
    """
    Bug fix v1.9: las stats deben ser size-invariant. La misma imagen a distintos tamaños
    debe dar el mismo preset (antes Hokusai daba edge=5% a 2000px pero 10.8% a 700px,
    cambiando el preset elegido).
    """
    lp = _load("laser_presets")
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    from PIL import Image  # noqa
    # Crear imagen con texturas finas a tamaño grande
    rng = np.random.default_rng(42)
    h, w = 1600, 1200
    rgb = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8)
    rec_big = lp.recommend_preset(rgb, material="acrylic_funsun_9060_back_engrave")

    # Misma imagen resized a 400px
    img = Image.fromarray(rgb)
    img_small = img.resize((400, 300), Image.Resampling.LANCZOS)
    rgb_small = np.array(img_small)
    rec_small = lp.recommend_preset(rgb_small, material="acrylic_funsun_9060_back_engrave")

    # Los presets deben coincidir (size-invariant)
    assert rec_big.preset_name == rec_small.preset_name, (
        f"size-variance: big={rec_big.preset_name} (edge={rec_big.stats.edge_density*100:.1f}%), "
        f"small={rec_small.preset_name} (edge={rec_small.stats.edge_density*100:.1f}%)"
    )


def test_cartoon_render_produces_solid_silhouette():
    """
    El preset cartoon debe convertir un dibujo (sujeto + fondo blanco) en silueta
    sólida: fondo PNG = 0 (no grabar), sujeto PNG = 255 (grabar como frost).
    """
    sys.path.insert(0, str(SCRIPT_SCRIPTS))
    import laser_target_match as ltm  # type: ignore
    lp = _load("laser_presets")
    p = lp.get_preset("cartoon_back_engrave")

    # Crear dibujo: fondo blanco grande + cuadrado oscuro en el medio
    gray = np.full((100, 100), 255, dtype=np.float64)
    gray[30:70, 30:70] = 80  # "sujeto" gris medio (cualquier color saturado)

    cand = ltm.Candidate(
        algorithm=p.algorithm, invert=p.invert,
        threshold=p.threshold, contrast=p.contrast, brightness=p.brightness,
        gamma=p.gamma, autocontrast=p.autocontrast, sharpen=p.sharpen,
    )
    out = ltm.render_candidate(gray, cand)

    # El cuadrado del sujeto (30:70, 30:70) debe ser 255 (grabar)
    subject_region = out[35:65, 35:65]
    assert (subject_region == 255).all(), (
        f"sujeto debe grabarse sólido, pero solo {(subject_region==255).mean()*100:.0f}% es 255"
    )
    # El fondo (esquinas) debe ser 0 (no grabar)
    bg_corners = out[:10, :10]
    assert (bg_corners == 0).all(), (
        f"fondo debe no grabarse, pero solo {(bg_corners==0).mean()*100:.0f}% es 0"
    )


def test_get_preset_by_name():
    lp = _load("laser_presets")
    p = lp.get_preset("portrait")
    assert p.name == "portrait"
    assert p.algorithm == "stucki_serpentine"


def test_get_preset_unknown_raises():
    lp = _load("laser_presets")
    with pytest.raises(KeyError):
        lp.get_preset("nonexistent_preset")


def test_list_presets_dict_serializable():
    lp = _load("laser_presets")
    items = lp.list_presets_dict()
    assert isinstance(items, list) and len(items) >= 5
    for item in items:
        assert "name" in item and "label" in item and "params" in item
        params = item["params"]
        for k in ("algorithm", "threshold", "contrast", "brightness", "gamma", "autocontrast", "sharpen", "invert"):
            assert k in params


def test_compute_image_stats_uniform_gray():
    """Imagen uniforme: std=0, edge_density~0."""
    lp = _load("laser_presets")
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    s = lp.compute_image_stats(arr)
    assert s.mean == pytest.approx(128.0, abs=0.5)
    assert s.std < 1.0
    assert s.edge_density < 0.01


def test_compute_image_stats_high_contrast_bimodal():
    """Imagen mitad negro / mitad blanco: extreme_ratio alto, edge_density localizado."""
    lp = _load("laser_presets")
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :, :] = 0
    arr[64:, :, :] = 255
    s = lp.compute_image_stats(arr)
    assert s.extreme_ratio > 0.95
    assert s.std > 100


def test_recommend_poster_for_bimodal_high_contrast():
    """Imagen bimodal (mucho negro + mucho blanco) + bordes finos densos → poster preset."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(0)
    # Simular gráfico: bandas alternantes de negro/blanco (alto edge_density por las bandas)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 8):
        arr[i:i+4, :, :] = 255  # bandas blancas finas → muchos bordes
    # Inyectar algo de texto-like ruido para variar
    for k in range(rng.integers(15, 25)):
        y, x = rng.integers(0, 250, size=2)
        v = int(rng.choice([0, 255]))
        arr[y:y+4, x:x+30, :] = v
    r = lp.recommend_preset(arr)
    assert r.preset_name in ("poster_back_engrave", "line_art"), f"got {r.preset_name}"


def test_recommend_scene_dark_for_dark_image():
    """Foto oscura realista: mean bajo (~40), pero std alto por sombras/midtones."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(1)
    # Gradiente oscuro 5..90 con variación de textura
    h, w = 128, 128
    base = np.linspace(5, 90, h, dtype=np.float64)[:, None] + np.zeros((h, w), dtype=np.float64)
    noise = rng.normal(0, 22, size=(h, w))
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    r = lp.recommend_preset(arr)
    assert r.preset_name == "scene_dark", f"got {r.preset_name} stats={r.stats}"
    assert "oscura" in r.reason.lower()


def test_recommend_scene_bright_for_bright_image():
    """Foto clara realista: mean alto (~200+), std alto por highlights/midtones."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(2)
    h, w = 128, 128
    base = np.linspace(155, 250, h, dtype=np.float64)[:, None] + np.zeros((h, w), dtype=np.float64)
    noise = rng.normal(0, 28, size=(h, w))
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    r = lp.recommend_preset(arr)
    assert r.preset_name == "scene_bright", f"got {r.preset_name} stats={r.stats}"


def test_recommend_photo_general_for_balanced_image():
    """Foto natural balanceada (gradientes, variación tonal moderada) → photo_general."""
    lp = _load("laser_presets")
    rng = np.random.default_rng(3)
    # Gradiente con ruido
    y_grad = np.linspace(80, 180, 128, dtype=np.float64)[:, None]
    x_grad = np.linspace(0, 30, 128, dtype=np.float64)[None, :]
    base = (y_grad + x_grad).clip(0, 255).astype(np.uint8)
    noise = rng.integers(-20, 21, size=(128, 128), dtype=np.int16)
    gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    r = lp.recommend_preset(arr)
    assert r.preset_name == "photo_general", f"got {r.preset_name}"


def test_recommendation_returns_image_stats():
    lp = _load("laser_presets")
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    r = lp.recommend_preset(arr)
    assert hasattr(r, "stats")
    assert hasattr(r.stats, "mean")
    assert hasattr(r.stats, "std")
    assert hasattr(r.stats, "extreme_ratio")
    assert hasattr(r.stats, "edge_density")


def test_preset_params_ranges_sane():
    """Sanity: cada preset tiene params dentro de rangos válidos."""
    lp = _load("laser_presets")
    for p in lp.ALL_PRESETS:
        assert 1 <= p.threshold <= 254, f"{p.name}: threshold {p.threshold}"
        assert 0.1 <= p.contrast <= 3.0, f"{p.name}: contrast {p.contrast}"
        assert -100 <= p.brightness <= 100, f"{p.name}: brightness {p.brightness}"
        assert 0.3 <= p.gamma <= 3.0, f"{p.name}: gamma {p.gamma}"
        assert 0 <= p.autocontrast <= 20, f"{p.name}: autocontrast {p.autocontrast}"
        assert 0 <= p.sharpen <= 300, f"{p.name}: sharpen {p.sharpen}"
