# Image AP — Laser Image Prep

Preparación de imágenes para grabado láser CO2. Convierte una foto a color en un
**PNG 1-bit listo para Pass-Through en LightBurn** (u otro CAM), con física del
láser cableada (DPI cap por spot, LUT por material, sharpen escalado al output
físico, simulación de grabado).

![Status](https://img.shields.io/badge/status-v1.2.0-success)
![Tests](https://img.shields.io/badge/tests-140%2B%2F140%2B-success)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![Installer](https://img.shields.io/badge/installer-.exe%20available-success)

---

## ¿Qué hace?

| Etapa | Qué hace |
|---|---|
| **Wizard web** (SvelteKit 5) | Subís foto → recortás → ajustes por material → preview en vivo → procesás full-res → simulás cómo se verá grabado → descargás PNG. |
| **Pipeline Python** | Preprocess (sauvola/niblack/grabcut/sam2) → LUT material → dither (40+ algoritmos) → PNG 1-bit. |
| **Scoring v1..v5** | Métricas para A/B entre params; v5 es **sin referencia** (HVS-MSE + spectral + tone post-LUT). |
| **Física CO2** | Spot del haz ↔ DPI útil; sharpen radius escalado al output mm × DPI; LUT inversa por material para compensar dot-gain. |
| **Calibración física** | Generador de step-wedge + fit de LUT desde foto del wedge grabado. |
| **Simulación** | Forward model: blur del spot + respuesta tonal del material para predecir cómo se verá fotografiado el grabado. |

---

## Quickstart (Windows)

### Opción 1 — Instalador `.exe` (recomendado, doble-clic)

1. Descargar `ImageAPLaser_Setup_v1.2.0.exe` desde
   [GitHub Releases](https://github.com/Vahlame/image-ap-sport-for-laser/releases/latest).
2. Doble-clic → asistente estándar Windows (idioma, licencia, carpeta destino).
3. Al terminar, Setup ejecuta automáticamente la instalación de Python, Node, y
   dependencias (via `winget`). Tarda ~15–30 min la primera vez.
4. Acceso directo "Image AP Laser" creado en el Escritorio → doble-clic para arrancar.

### Opción 2 — Manual (developers / sin Internet en instalación)

```powershell
git clone https://github.com/Vahlame/image-ap-sport-for-laser
cd image-ap-sport-for-laser
Setup_LaserApp.bat       # configura todo: winget → Python/Node → venv → deps → web build
Iniciar_Laser_App.bat    # arranca uvicorn + abre el wizard en el navegador
```

### Manual (dos terminales)

```powershell
# Terminal A — backend
.\.venv312\Scripts\Activate.ps1
uvicorn scripts.api_server:app --host 127.0.0.1 --port 18765

# Terminal B — wizard dev (opcional; sólo si querés hot-reload de la UI)
cd web
npm run dev
```

Abrí `http://127.0.0.1:18765/app/` (build static servido por FastAPI) o
`http://localhost:5173` (modo dev SvelteKit).

---

## Workflow operador (5 pasos)

1. **Subir** — drag-drop una foto (PNG/JPG/WebP, hasta 8000×8000 px).
2. **Recortar** — encuadre con `cropper.js`.
3. **Ajustes** — el wizard usa **preset `auto`** por defecto: detecta el tipo de imagen
   (foto general, retrato, escena oscura/clara, poster, line art) y aplica los params
   adecuados. Si querés tunear, cambiá a `manual` y movés sliders. Preview en vivo
   (~300 ms debounce).
4. **Resultado full-res** — PNG 1-bit + **simulación del grabado** (cómo se verá
   fotografiado en acrílico frost o wood burn).
5. **Descargar** — `laser_ready_<material>_<algoritmo>.png` + checklist pre-grabado
   (mirror back-engrave, interval `25.4/DPI`, Pass-Through, 9–12% potencia acrílico).

### Presets curados (v1.1.0+)

El backend mantiene un catálogo de presets validados experimentalmente. El detector
elige el mejor según estadísticos básicos de la imagen (luminancia media, contraste,
distribución bimodal, densidad de bordes):

| Preset | Cuándo se aplica | Algoritmo | Polaridad |
|---|---|---|---|
| **photo_general** | Foto natural balanceada (default fallback) | Jarvis serpentine | invertido (positive en madera) |
| **portrait** | Cara/animal en primer plano | Stucki serpentine | invertido |
| **scene_dark** | Foto oscura con highlights brillantes | Jarvis serpentine | normal (engraba highlights) |
| **scene_bright** | Foto muy clara (cielo, nieve, blanco) | Jarvis serpentine | invertido |
| **poster_back_engrave** | Gráfico bimodal alto contraste | Floyd + sauvola | invertido (frost sobre fondo oscuro) |
| **line_art** | Líneas finas / texto / vector | Threshold sin dither | invertido |

Si querés saltar el auto y elegir explícitamente, pasá `preset: <nombre>` en el JSON
del request (o seleccionalo en el wizard).

---

## CLI avanzado (investigación / sweeps)

Para barridos masivos, refinaciones, calibración. Documentado completo en
[`docs/USAGE.md`](docs/USAGE.md).

```powershell
# Single render con params concretos
python scripts\laser_target_match.py `
  --input foto.jpg --target referencia.png `
  --out runs\manual --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 0 --n 1 --workers 1 `
  --material acrylic_back_engrave `
  --output-mm-short 100 --output-dpi 169 --sharpen-radius-mm 0.10

# Generar tira step-wedge para calibrar tu Funsun
python scripts\laser_calibration_wedge.py `
  --out wedge_acrylic.png --material acrylic_back_engrave `
  --steps 16 --square-mm 10 --dpi 169 --dither blue_noise_vac32

# Tras grabar + fotografiar la tira, ajustar LUT
python scripts\laser_calibration_fit.py `
  --photo wedge_grabado.jpg `
  --wedge-meta wedge_acrylic_meta.json `
  --out presets\materials\acrylic_funsun_calibrated.npy
```

---

## Endpoints REST

`scripts/api_server.py` expone:

- `GET  /api/health` — liveness + estado modelo LPIPS + CUDA.
- `GET  /api/materials` — builtins (`acrylic_back_engrave`, `wood_generic`) + custom de
  `presets/materials/`.
- `GET  /api/algorithms` — 40+ algoritmos agrupados por familia.
- `POST /api/preview` — multipart imagen + `params_json`, fuerza `max_side=400`.
- `POST /api/process` — full-res, devuelve PNG 1-bit con headers (`X-Process-Time-Ms`,
  `X-Output-Width`, `X-White-Ratio`, `X-Sharpen-Radius-Px`, `X-Material`).
- `POST /api/simulate` — multipart imagen 1-bit + material → PNG simulado del grabado.

Ejemplo desde Python:

```python
import requests, json
with open('foto.jpg', 'rb') as f:
    r = requests.post(
        'http://127.0.0.1:18765/api/process',
        files={'image': f},
        data={'params_json': json.dumps({
            'material': 'acrylic_back_engrave',
            'algorithm': 'floyd', 'threshold': 75, 'contrast': 1.0,
            'brightness': 10, 'gamma': 1.2, 'invert': True,
            'preprocess_mode': 'sauvola',
            'output_mm_short': 100, 'output_dpi': 169,
        })}
    )
open('laser_ready.png', 'wb').write(r.content)
```

---

## Arquitectura

```
web/                       SvelteKit 5 wizard (build static)
  src/routes/+page.svelte  5-step wizard
  src/lib/apiClient.ts     cliente HTTP tipado
  src/lib/components/      CropStage.svelte (cropper.js)

scripts/                   motor Python
  api_server.py            FastAPI (port 18765 default)
  laser_target_match.py    motor principal (40+ algos, 9 preprocess, scoring v1..v5)
  laser_scoring.py         métricas v1..v5
  laser_physics.py         MaterialProfile, DPI cap, scaled USM
  laser_blue_noise.py      void-and-cluster Ulichney 1993
  laser_calibration_wedge.py  generador step-wedge
  laser_calibration_fit.py    fit LUT desde foto del wedge
  laser_simulator.py       forward model (spot blur + apariencia material)
  laser_runtime_env.py     LPIPS device, VRAM caps, HF token sync

tests/                     pytest, 115+ tests
assets/blue_noise_*.npy    matrices VAC cacheadas
presets/materials/         JSONs de materiales calibrados (post-calibración)
docs/                      USAGE.md, IMPROVEMENT_LOG.md, planes
runs/                      experimentos (gitignored)
```

---

## Estado del proyecto

- **v1.0.0** (2026-05-15): primera versión funcional verificada con dos imágenes reales
  (poster Agricultor: pixel-match 69.74% vs target ImagR; retrato Picsum: visualmente
  excelente con Jarvis serpentine).
- **115/115 tests passing** (`pytest -m "not network"`). `svelte-check` clean.
- **CUDA opcional**: LPIPS acelera con GPU; sin CUDA cae a CPU automáticamente.

---

## Licencia

**GPL-3.0-or-later** (copyleft). Ver [`LICENSE`](LICENSE).

En claro:

- ✅ Podés usarlo, modificarlo y distribuirlo libremente, incluso comercialmente.
- ✅ Podés contribuir vía PR (ver [`CONTRIBUTING.md`](CONTRIBUTING.md) y
  añadirte en [`AUTHORS.md`](AUTHORS.md)).
- 🔒 Si distribuís una versión modificada, **debe seguir siendo GPL-3.0**: no se permite
  re-licenciarla bajo términos más restrictivos ni cerrarla source-closed.
- 📝 Debés preservar los avisos de copyright y créditos de los autores.

Las matemáticas de los algoritmos de dithering son dominio público (Floyd-Steinberg
1976, Jarvis 1976, Stucki 1981, Atkinson 1986, Burkes 1988, Sierra 1989,
Ulichney void-and-cluster 1993). Implementación propia desde los papers; no se copió
código propietario.

---

## Roadmap (más allá de v1.0)

- **Calibración física real**: workflow listo, requiere que el usuario grabe el
  step-wedge en su láser + fotografíe. Después la LUT custom queda permanente.
- **WebSocket** para progreso de sweeps largos.
- **Métricas perceptuales mejoradas**: `spectral_radial_penalty` con high-pass previo
  para distinguir dither de macro-estructura.
- **PyInstaller .exe**: empaquetado para distribución a operadores sin Python instalado
  (hoy requiere venv + npm; el `.bat` los wrappea).

Ver [`docs/IMPROVEMENT_LOG.md`](docs/IMPROVEMENT_LOG.md) para historial técnico y
[`docs/USAGE.md`](docs/USAGE.md) para guía operador completa.
