# Image AP — Laser Image Prep

Preparación de imágenes para grabado láser CO2. Convierte una foto a color en un
**PNG 1-bit listo para Pass-Through en LightBurn** (u otro CAM), con física del
láser cableada (DPI cap por spot, LUT por material, sharpen escalado al output
físico, simulación de grabado).

![Status](https://img.shields.io/badge/status-v2.3.0-success)
![Tests](https://img.shields.io/badge/tests-186%2F186-success)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-GPL--3.0%20%7C%7C%20Commercial-blueviolet)
![Installer](https://img.shields.io/badge/installer-.exe%20%2B%20zip-success)
![Mode](https://img.shields.io/badge/mode-Express-brightgreen)
![Presets](https://img.shields.io/badge/presets-9-blueviolet)

> ⚠️ **DUAL LICENSING** — Este proyecto usa licenciamiento dual.
> **GPL-3.0-or-later** (open-source, copyleft) para uso personal/educativo/OSS, o
> **Licencia Comercial** (royalty/% acordado) para producto closed-source / SaaS /
> distribución comercial. Mirá [`LICENSE`](LICENSE) y [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md).

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

1. Descargar `ImageAPLaser_Setup_v2.0.0.exe` desde
   [GitHub Releases](https://github.com/Vahlame/image-ap-sport-for-laser/releases/latest).
2. **Si Windows SmartScreen bloquea con "Windows protegió su PC"**: ver sección
   [⚠️ Windows SmartScreen](#-windows-smartscreen-warning) abajo.
3. Doble-clic → asistente estándar Windows (idioma, licencia, carpeta destino).
4. Al terminar, Setup ejecuta automáticamente la instalación de Python, Node, y
   dependencias (via `winget`). Tarda ~15–30 min la primera vez.
5. Acceso directo "Image AP Laser" creado en el Escritorio → doble-clic para arrancar.

### Opción 2 — ZIP portable (sin warning de SmartScreen)

Si querés evitar el warning de SmartScreen del `.exe`:

1. Descargar `ImageAPLaser_Portable_v2.0.0.zip` desde
   [GitHub Releases](https://github.com/Vahlame/image-ap-sport-for-laser/releases/latest).
2. Click derecho → **Propiedades → Desbloquear** (Windows "marca" archivos bajados de Internet).
3. Extraer el ZIP a `C:\ImageAPLaser` (o donde prefieras).
4. Doble-clic en `Setup_LaserApp.bat` → mismo workflow que el .exe pero sin warning.
5. Después, `Iniciar_Laser_App.bat` para usar la app.

### Opción 3 — Git clone (developers)

```powershell
git clone https://github.com/Vahlame/image-ap-sport-for-laser
cd image-ap-sport-for-laser
Setup_LaserApp.bat       # configura todo: winget → Python/Node → venv → deps → web build
Iniciar_Laser_App.bat    # arranca uvicorn + abre el wizard en el navegador
```

---

## ⚠️ Windows SmartScreen Warning

Cuando descargues el `.exe` por primera vez, Windows va a mostrar:

> "Windows protegió su PC"
> "SmartScreen de Microsoft Defender evitó que se iniciara una aplicación desconocida"
> "Editor: Editor desconocido"

**Esto NO significa que el .exe sea malicioso** — es el comportamiento normal de Windows
para cualquier ejecutable que no esté firmado con un certificado Authenticode de pago
(Verisign, DigiCert, etc., ~$400-700 USD/año) y que no haya acumulado "reputación" con
suficientes descargas (~10,000+). Pasa con Git, VLC, Blender, Audacity, etc. en sus
primeras releases.

### Cómo evitarlo

**Opción A — Bypass del warning (rápido)**:
1. En el diálogo "Windows protegió su PC", hacer clic en **"Más información"**
   (link chico debajo del texto azul).
2. Aparece el botón **"Ejecutar de todas formas"** → clic.
3. El installer arranca normalmente.

**Opción B — Usar el ZIP portable**:
Descargar `ImageAPLaser_Portable_v2.0.0.zip` en vez del `.exe`. Los ZIP **no
generan warning de SmartScreen**. Click derecho → Propiedades → Desbloquear, extraer
y correr `Setup_LaserApp.bat` igual que el .exe.

**Opción C — Para mantainers/contribuidores**: ver
[`docs/CODE_SIGNING.md`](docs/CODE_SIGNING.md) sobre cómo aplicar a
[SignPath.io](https://signpath.io/) (firma gratis para proyectos OSS confirmados
como este) o comprar un certificado comercial.

### Cómo verificar que el .exe es legítimo

Si querés validar la integridad del descargado contra el oficial:

```powershell
# El SHA256 del v2.0.0 oficial está publicado en la Release de GitHub
Get-FileHash -Algorithm SHA256 ImageAPLaser_Setup_v2.0.0.exe
# Debería coincidir con: 803b7f5987359b2a89a67b19dce5be809eac42efa97dd9d0524d387c4f264505
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

## Modo Express (v1.3.0) ⚡ — recomendado

El flujo más simple posible: subís una foto y la app la procesa con tu configuración guardada.

1. Primera vez: abrís `⚙ Mi configuración` en el wizard y guardás:
   - **Material**: ej. `acrylic_funsun_9060_back_engrave` (tu máquina).
   - **Lado corto (mm)**: tamaño físico de la pieza.
   - **DPI**: típicamente el `default_dpi` del material (115 para Funsun 2.5").
2. Después arrastrás cualquier foto al dropzone Express.
3. La app:
   - Auto-detecta el tipo de imagen (foto/retrato/escena/poster/line-art).
   - Aplica el preset adecuado con HQ refinement (25 candidatos sobol + scoring v5).
   - Devuelve PNG laser-ready en 2–3 min.
   - Muestra **`Configuración recomendada para LightBurn`** lista para copiar
     (DPI, interval mm, power %, speed mm/s, MirrorX, Pass-Through, invertir).

## Modo Manual (5 pasos) 🛠

Para fotos donde querés tunear cada slider:

1. **Subir** — drag-drop una foto (PNG/JPG/WebP, hasta 8000×8000 px).
2. **Recortar** — encuadre con `cropper.js`.
3. **Ajustes** — preset `auto` o manual. Material, mm, DPI, algoritmo, contrast/brightness/
   gamma/sharpen. Preview en vivo (~300 ms debounce).
4. **Resultado full-res** — PNG 1-bit + **simulación del grabado** + panel `Configuración LightBurn`.
5. **Descargar** — `laser_ready_<material>_<algoritmo>.png` + checklist pre-grabado.

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

## Licencia — Dual Licensing (v2.3+)

Este proyecto usa **dual licensing**: elegís el camino que se ajusta a tu caso de uso.

### 🟢 Camino A — GPL-3.0-or-later (gratis, OSS / personal)

**Para**: uso personal, educativo, investigación, hobbyistas, proyectos OSS.

- ✅ Podés usarlo, modificarlo, distribuirlo y **usarlo en tu taller para grabar
  para clientes** (el grabado es el producto, no el software).
- ✅ Podés contribuir vía PR (ver [`CONTRIBUTING.md`](CONTRIBUTING.md)).
- 🔒 **Cualquier trabajo derivado distribuido** DEBE seguir siendo GPL-3.0
  (copyleft). No se permite cerrarlo source-closed ni redistribuirlo bajo otra
  licencia.
- 📝 Debés preservar los avisos de copyright + agregar tus cambios al
  `CHANGELOG.md` si distribuís un fork.

Ver el texto legal completo en [`LICENSE`](LICENSE).

### 🔴 Camino B — Commercial License (paga, closed-source / SaaS / producto)

**REQUERIDO si**:
- Vas a vender un **producto físico** (ej. máquina láser con software embebido).
- Vas a ofrecer un **SaaS / servicio web** que use este software internamente.
- Vas a integrar el código en un producto **closed-source** sin liberar el tuyo
  bajo GPL-3.0.
- Tu organización tiene **políticas anti-GPL** (común en corporaciones).
- Vas a **distribuir comercialmente** el software o un trabajo derivado.

**Cómo funciona**:
- Negociás términos con el copyright holder (Vahlame): royalty %, fee anual,
  o pago único según volumen.
- Templates de términos típicos: **3–12% royalty** sobre ventas/MRR, o
  **USD 500–5,000/año** según tamaño de empresa.
- Acuerdo escrito firmado por ambas partes antes de uso comercial.

Ver detalles completos y cómo contactar en [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md).

> **⚠️ AVISO**: usar el software comercialmente **SIN** acordar la licencia
> comercial Y **SIN** liberar tu código completo bajo GPL-3.0 constituye
> violación de copyright. Por favor consultá antes de empezar.

### 🤝 Casos comunes — qué licencia te aplica

| Caso | Licencia |
|---|---|
| Lo uso en mi taller para grabar piezas para mis clientes | **GPL-3.0** ✅ (gratis) |
| Tengo un sitio web donde clientes suben fotos y reciben PNGs procesados | **Commercial** 🔴 |
| Lo modifiqué para mi uso interno en mi empresa de < 10 personas | **GPL-3.0** ✅ si no distribuís el binary; **Commercial** 🔴 si lo distribuís |
| Vendo una máquina láser con este software pre-instalado | **Commercial** 🔴 |
| Hago un fork open-source con mejoras y lo subo a GitHub bajo GPL-3.0 | **GPL-3.0** ✅ |
| Doy un curso pago de grabado láser usando este software como herramienta | **GPL-3.0** ✅ (no es distribución del software) |
| Vendo plantillas / archivos PNG generados con el software | **GPL-3.0** ✅ (el output PNG no es trabajo derivado del software) |

### 📐 Algoritmos científicos vs implementación

Las **matemáticas** de los algoritmos de dithering están en dominio público
(Floyd-Steinberg 1976, Jarvis 1976, Stucki 1981, Atkinson 1986, Burkes 1988,
Sierra 1989, Ulichney void-and-cluster 1993). Cualquiera puede implementar
estos algoritmos desde los papers originales — eso no infringe nuestro copyright.

Lo que **SÍ** está protegido por copyright de Vahlame + colaboradores:
- La **implementación específica** en Python (incluyendo optimizaciones numba JIT).
- El **auto-detector con 5 reglas** + presets curados (`photo_high_detail`,
  `cartoon_back_engrave`, etc.) — validación empírica original del proyecto.
- El **score v5** (HVS-MSE + spectral + edge preservation + multi-scale tone).
- El **wizard SvelteKit** y toda la integración FastAPI.
- El **plain_region_simplification** + auto-mirror + integraciones LightBurn.

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
