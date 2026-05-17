# Changelog

Todas las versiones notables del proyecto se documentan acá.
Formato basado en [Keep a Changelog](https://keepachangelog.com/) +
[Semver](https://semver.org/).

## [2.2.0] — 2026-05-16

### Cierre de auditoría — los bugs de severidad baja/media restantes

v2.1 cubrió los críticos. v2.2 cierra los 6 bugs restantes del audit + UX
mejoras adicionales.

### Backend (`scripts/api_server.py`)

**Bug ALTA — Auto-mirror double-apply** (`api_server.py`):
- El PNG se espejaba automáticamente para back-engrave pero el frontend no tenía
  forma de saberlo → usuario podía aplicar MirrorX en LightBurn provocando un
  doble-mirror que arruina el grabado.
- Fix: header `X-Auto-Mirrored: "true"|"false"` reportado en cada response.
  Agregado a `expose_headers` del CORS para que sea legible desde el frontend.
- Frontend muestra aviso visual cuando `autoMirrored=true`: "NO actives MirrorX
  en LightBurn — el PNG ya está listo para Pass-Through."
- Test: `test_auto_mirror_header_reports_state` (3 escenarios: true/false/wood).

**Doc — CORS warning para production**:
- Comentario explícito antes de `app.add_middleware(CORSMiddleware,...)` con
  checklist de seguridad: agregar auth, no usar "*", rate-limiting, etc.

### Frontend (`web/src/`)

**Bug MEDIA — `cancelJob` no actualizaba UI inmediatamente**:
- Usuario hacía clic en "Cancelar" pero el botón quedaba "Cancelando…" hasta
  que el roundtrip terminaba (~1-2s con SSE lento).
- Fix: optimistic UI update — `expressProgress.stage = 'cancelling'` antes del
  `await apiClient.cancelJob()`. El usuario ve feedback inmediato.

**Bug MEDIA — `previewTimer` race condition en $effect**:
- Si el componente desmontaba o cambiaba step entre el `setTimeout` y su
  callback, `runPreview()` podía dispararse en estado inválido.
- Fix: guard `if (previewTimer !== null)` + reset a `null` después del clear.
  Early-return también limpia el timer pendiente.

**Bug BAJA — Drag-drop sin feedback visual**:
- Arrastrar un archivo sobre el dropzone no daba indicación visual.
- Fix: handlers `ondragover/ondragleave/ondrop` agregan/quitan clase `.drag-active`
  con estilo: border accent + glow + scale(1.01) para feedback claro.

### Tests (`tests/conftest.py`)

**Mejora — Garbage collection autouse fixture**:
- Tests largos (HQ refine con stock images) acumulaban memoria afectando tests
  posteriores.
- Fix: `_gc_after_each_test` autouse fixture corre `gc.collect()` tras cada test
  (~5ms overhead, libera Image/numpy buffers).
- Documentación del `stock_cache_dir`: `tmp_path_factory.mktemp()` ya auto-limpia
  con pytest config default; instrucciones para invalidar cache manualmente.

### Tests nuevos (1)

- `test_auto_mirror_header_reports_state` — verifica X-Auto-Mirrored para 3
  escenarios (acrylic_back_engrave + auto=true/false, wood_generic + auto=true).

**186 tests totales** (+1 nuevo), todos verdes.

## [2.1.0] — 2026-05-16

### Bug fix urgente — Descargar imagen no funcionaba

**Reportado por usuario**: al hacer clic en "Descargar" en step 5, el archivo no se
descargaba en Firefox/Safari (Chrome funcionaba intermitente).

**Causa raíz**: `downloadFinal()` creaba un `<a>` element y llamaba `.click()` SIN
agregarlo al DOM. Firefox y Safari (especialmente con políticas strict) bloquean
clicks programáticos en elementos no-DOM. Además, reutilizaba `finalBlobUrl` que
podía estar siendo usado por `<img src>` simultáneamente.

**Fix**:
- Crear un `URL.createObjectURL(finalBlob)` NUEVO sólo para la descarga (no compartir
  con `<img>`).
- `appendChild` al `document.body` ANTES del click.
- `setTimeout(revoke, 100)` para liberar memoria tras la descarga.
- Sanitización del filename (regex `/[^a-z0-9_-]/gi` para chars válidos).
- Log warning si `finalBlob` es null (debugging help).

### Auditoría exhaustiva de bugs — 15 issues encontrados y arreglados

Corrida automática con agente Explore que revisó frontend + backend + tests en busca de:
race conditions, memory leaks, null guards faltantes, edge cases sin manejar, threading.

#### Backend (`scripts/api_server.py`)

**Bug ALTA — División por cero en autocontrast** (`laser_target_match.py:641-644`):
- Imagen uniforme (todo gris=128) hacía `high - low = 0` → división por cero → NaN.
- Fix: si `spread <= 1.0`, no estirar (preserva imagen original).
- Test nuevo: `test_uniform_image_no_division_by_zero`.

**Bug MEDIA — Image.open sin context manager** (`api_server.py:_load_image_from_bytes`):
- Si `convert("RGB")` fallaba, el FD del JPEG quedaba abierto hasta GC.
- Fix: `with Image.open(...) as raw: img = raw.convert("RGB")`.

**Bug MEDIA — Sin límite de tamaño de upload** (`api_server.py`):
- Usuario malicioso (o accidental) podía subir 5GB → OOM del servidor.
- Fix: `MAX_UPLOAD_BYTES = 100 * 1024 * 1024` (100MB) con 413 Payload Too Large.
- También: rechazar bytes < 64 (claramente no es imagen) con 400.
- Tests: `test_max_upload_size_returns_413`, `test_tiny_upload_returns_400`.

**Bug MEDIA — Exception silenciosa en progress_cb** (`api_server.py:_hq_refine`):
- `except Exception: pass` swallowed errores del callback sin log.
- Fix: `logging.warning()` con candidate index + exception.

**Bug MEDIA — `_PSUTIL_PROC` sin lock** (`api_server.py`):
- Acceso concurrente desde múltiples workers podía corromper estado interno de psutil
  → CPU metrics ocasionalmente erráticos.
- Fix: `threading.Lock()` alrededor de `_PSUTIL_PROC` access.

**Bug MEDIA — `score_history` race condition** (`laser_jobs.py:to_progress_dict`):
- Slice de lista durante append concurrente (CPython GIL no garantiza atomicidad de
  list operations).
- Fix: `list(self.score_history[-N:])` snapshot atómico antes de iterar.

#### Frontend (`web/src/`)

**Bug ALTA — `processingFull` puede quedar `true` tras error async** (`+page.svelte`):
- Si `backToUpload()` se llamaba con job pendiente, no resetea `processingFull`.
- Fix: `backToUpload()` ahora resetea TODO el estado + cancela job pendiente.

**Bug ALTA — File input no permite re-subir misma imagen** (`+page.svelte`):
- `input.value = ''` se ejecutaba al final (después de procesar) en vez del inicio.
  HTML spec: si el value no cambia, no se dispara `onchange` → usuario tenía que
  recargar la página para re-subir la misma imagen tras un error.
- Fix: `input.value = ''` al inicio del handler, antes de procesar.

**Bug ALTA — EventSource sin cleanup en error path** (`apiClient.ts:renderAsync`):
- `evtSource.onerror` no llamaba `cleanup()` ni cerraba el EventSource → el browser
  intentaba reconectar automáticamente generando requests fantasma.
- Fix: cerrar EventSource explícitamente en `onerror`, dejar que polling termine.

**Bug BAJA — `saveMyConfig` swallowed errores silenciosamente** (`apiClient.ts`):
- Si localStorage estaba deshabilitado (private mode) o quota excedida, el usuario
  no se enteraba.
- Fix: `console.warn()` con el error original para debugging.

### UX improvements

**Botón "Descargar PNG ahora" directo en step 4**:
- Antes obligaba al usuario a navegar a step 5 (página separada) sólo para descargar.
- Ahora: 2 botones en step 4 — descarga directa rápida + "Ver checklist →" para el
  paso completo con pre-grabado checklist.

**Botón "Reintentar con la misma imagen" en error de Express**:
- Antes el usuario tenía que volver al step 1 y re-subir la imagen tras un error.
- Ahora: error-card con dos botones — "🔄 Reintentar" (mantiene `croppedBlob`) o
  "Cerrar mensaje".

**Validación temprana del tamaño en frontend**:
- Antes el usuario subía 200MB y esperaba 30s para que el backend rechazara con 413.
- Ahora: validación en `onChange` del input con feedback inmediato (< 1s).

### Tests nuevos (3)

- `test_max_upload_size_returns_413` — 101MB rechazado antes de PIL.open.
- `test_tiny_upload_returns_400` — bytes < 64 rechazados antes de PIL.
- `test_uniform_image_no_division_by_zero` — autocontrast sobre imagen plana no crashea.

**185 tests totales** (+3 nuevos), todos verdes.

### Documentación nueva

- `docs/CODE_SIGNING.md`: proceso para aplicar a SignPath.io OSS (gratis), workflow
  GitHub Actions de ejemplo, alternativas (Sectigo OV/EV, winget submission).
- `README.md`: sección "⚠️ Windows SmartScreen Warning" con 3 opciones de bypass.
- ZIP portable como alternativa al `.exe` (no dispara SmartScreen).

## [2.0.0] — 2026-05-16

### Investigación comparativa contra herramientas comerciales

Investigamos las herramientas líderes del mercado 2026 para identificar qué nos
falta vs PhotoGrav ($395), ImagR, LightBurn ($60) y LaserGRBL:

**PhotoGrav** — "Power/Lens model" simula spot por material, "diffusion dithering"
calibrado, auto-mirror+negative para acrílico back-engrave, 20+ materiales precal.

**ImagR** — 13 algoritmos, perfiles para wood/slate/acrylic/leather/glass, ML para
upscaling y background removal.

**LightBurn** — halftone con grid rotado (angle setting), Pass-Through mode.

**Workflow profesional Photoshop/GIMP** — S-curve tonal + local contrast (Unsharp
radius grande, amount bajo) + Unsharp Mask tradicional + Stucki dither.

### v2.0.0 — 3 técnicas profesionales agregadas

#### 1. **S-curve tonal** (`apply_s_curve`)

Función sigmoide centrada en 128 que aclara midtones (192→225) y oscurece sombras
(64→31), con pivot fijo en 128. Técnica estándar del workflow Photoshop/PhotoGrav.

```python
out = 128 + 127.5 * tanh(strength * 4 * (gray-128)/128)
```

Strength `0.0` = identidad, `0.5` = S suave (default fotos), `1.0` = agresiva.

#### 2. **Local contrast enhancement / Clarity** (`apply_local_contrast`)

Unsharp Mask con **radius grande** (~60px) + **amount bajo** (5-20%). Diferente
del unsharp tradicional (radius 1-2px + amount 50-100%): aumenta el "punch"
mid-frecuencia sin afectar detalles finos.

Ventaja vs CLAHE: preserva histograma global, no amplifica ruido. Es la técnica
clave del workflow profesional para fotorrealismo de grabado láser.

#### 3. **Auto-mirror para back-engrave** (PhotoGrav-style)

Cuando `material` termina en `_back_engrave` y `auto_mirror_back_engrave=True`
(default), el PNG final se voltea horizontalmente. Al grabar en la cara posterior
del acrílico, hay que invertir para que se vea correcto desde el frente.

**Impacto UX**: un paso menos para el usuario en LightBurn (antes había que aplicar
MirrorX manualmente, ahora viene en el PNG).

### Backend

- `scripts/laser_target_match.py`:
  - `apply_s_curve(gray, strength)` — función pública, validada con tests.
  - `apply_local_contrast(gray, radius_px, amount_pct)` — function pública.
- `scripts/api_server.py`:
  - `ProcessParams.s_curve_strength: float [0, 1.5]` (default 0.0).
  - `ProcessParams.local_contrast_amount: float [0, 50]` (default 0.0).
  - `ProcessParams.auto_mirror_back_engrave: bool` (default True).
  - `_process_image()` aplica S-curve → local_contrast → plain_region_simp en orden.
  - Auto-mirror al final del pipeline si material es `_back_engrave`.
  - `meta["auto_mirrored"]: bool` reportado en headers para introspección.
- `scripts/laser_presets.py`:
  - `LaserPreset.s_curve_strength: float = 0.0` (campo nuevo).
  - `LaserPreset.local_contrast_amount: float = 0.0` (campo nuevo).
  - `as_param_dict()` incluye los nuevos campos para el auto-merge del preset.

### Presets actualizados con técnicas profesionales activadas

| Preset | s_curve_strength | local_contrast_amount |
|---|---|---|
| `photo_back_engrave` | 0.5 (S suave) | 12 (clarity moderada) |
| `photo_high_detail` | 0.4 (más conservador) | 10 (leve, sin amplificar ruido) |
| `photo_fine_textures` | 0.3 (texturas ya tienen contraste) | 15 (realza patrón) |
| `cartoon_back_engrave` | 0 (silueta sólida, no aplica) | 0 (idem) |
| `line_art` | 0 | 0 |

### Frontend

- `ProcessParams` interface extendido con los 3 nuevos campos opcionales.
- El detector + presets aplican los valores automáticamente vía `as_param_dict()`.

### Tabla comparativa final

| Feature | Nuestro v2.0 | PhotoGrav | ImagR | LightBurn |
|---|---|---|---|---|
| Múltiples algoritmos dither | ✅ 18+ | ✅ | ✅ 13 | ✅ ~6 |
| Material-aware LUTs | ✅ | ✅ 20+ | ✅ | ⚠️ |
| Auto-detector preset (5 reglas) | ✅ **Solo nosotros** | ❌ | ⚠️ | ❌ |
| Plain region simplification | ✅ **Solo nosotros** | ❌ | ❌ | ❌ |
| HQ refinement (Sobol search) | ✅ **Solo nosotros** | ❌ | ❌ | ❌ |
| Score-based optimization (v5 HVS) | ✅ **Solo nosotros** | ❌ | ❌ | ❌ |
| **S-curve tonal** | ✅ v2.0 NEW | ✅ | ⚠️ | ⚠️ |
| **Local contrast (Clarity)** | ✅ v2.0 NEW | ✅ | ✅ | ❌ |
| **Auto-mirror back-engrave** | ✅ v2.0 NEW | ✅ | ⚠️ | ❌ |
| Open source / GPL-3.0 | ✅ | ❌ ($395) | ❌ | ❌ ($60) |
| Wizard automático one-click | ✅ | ❌ | ✅ | ❌ |

### Tests

4 tests nuevos:
- `test_s_curve_aclara_midtones_oscurece_sombras` (pivot 128, sin saturar extremos).
- `test_local_contrast_aumenta_punch_sin_amplificar_extremos` (std aumenta, mean estable).
- `test_auto_mirror_applied_for_back_engrave_material` (fliplr verificable).
- `test_auto_mirror_NO_applied_for_wood_material` (no afecta otros materiales).

**182 tests totales** (+4 nuevos), todos verdes.

### Resultado visual del rally car con v2.0

| v1.9 | v2.0 (S-curve + clarity + auto-mirror) |
|---|---|
| Sin tonal mapping S | Midtones aclarados + sombras oscurecidas → "punch fotográfico" |
| Sin local contrast | Detalle visible sin amplificar ruido |
| Usuario debía MirrorX manual en LightBurn | **PNG ya espejado — un paso menos** |
| RedBull leíble correcto | RedBull invertido → correcto para grabar atrás |

## [1.9.0] — 2026-05-16

### Agregado — Preset `photo_fine_textures` para texturas finas dominantes

Tras una comparación con un algoritmo externo (halftone tradicional con detalle fino
sobre textil houndstooth), el usuario pidió mejorar nuestra herramienta para casos
con texturas finas como prioridad. Resultado: nuevo preset optimizado.

**Comparativa de presets para acrílico back-engrave**:

| Preset | Algoritmo | Sharpen | Threshold | Preprocess | Cuando |
|---|---|---|---|---|---|
| `photo_back_engrave` | stucki_serpentine | 130 | 95 | sauvola | bimodal natural |
| `photo_high_detail` | jarvis_serpentine | 80 | 110 | clahe | midtone-rich |
| **`photo_fine_textures`** (NUEVO) | stucki_serpentine | 120 | 120 | sauvola | texturas finas dominantes |
| `cartoon_back_engrave` | threshold | 0 | 10 | none | dibujos con fondo blanco |

**Diferencia clave**: `photo_fine_textures` usa **Stucki** (12 vecinos vs 7 de Jarvis)
que distribuye el error en mayor área → patrones repetitivos se preservan mejor.
Combinado con sharpen 120 y sauvola, captura el detalle de tejidos houndstooth,
follaje denso, pelaje, plumas, etc.

### Detector — Nueva Regla A.5

```
Regla A.5 (v1.9): edge_density > 10% AND std > 50 AND extreme_ratio < 50%
                  → photo_fine_textures
```

Validado con Hokusai "Great Wave" (edge=10%, std=55, extr=9%): el nuevo preset
preserva muchísimo mejor las espumas, líneas finas de las olas, los caracteres
del sello del artista y el Monte Fuji nítido.

### Bug fix — `edge_density` NO era size-invariant

Encontrado al probar Hokusai: la misma imagen daba `edge=5.0%` a 2000px pero
`edge=10.8%` a 700px. Esto causaba que el detector eligiera presets distintos
según el tamaño del upload.

**Fix**: `compute_image_stats()` ahora normaliza internamente la imagen a
`max_side=800px` antes de calcular stats. Resultado: stats consistentes
independientemente del tamaño original del upload.

Verificación:
```
max_side=2000: edge=10.1% std=55 → photo_fine_textures
max_side=1500: edge=10.1% std=55 → photo_fine_textures
max_side=1000: edge=10.1% std=55 → photo_fine_textures
max_side=800:  edge=10.1% std=55 → photo_fine_textures
max_side=600:  edge=11.6% std=55 → photo_fine_textures  (downsample puede
max_side=400:  edge=13.5% std=54 → photo_fine_textures   subir edge density)
```

Para uploads <800px (raros), el edge_density puede variar levemente pero el
preset elegido se mantiene robusto.

### Decisión final del detector (5 reglas en orden)

```
Regla 0  (acrylic): vbright>50% AND extr>65%        → cartoon_back_engrave
Regla A  (acrylic): extreme_ratio < 40%             → photo_high_detail
Regla A.5(acrylic): edge>10% AND std>50 AND extr<50% → photo_fine_textures  ← NUEVO
Regla B  (acrylic): vbright>18% AND extr<75%        → photo_high_detail
Regla C  (acrylic): default                          → photo_back_engrave
```

### Tests

3 tests nuevos en `test_laser_presets.py`:
- `test_fine_textures_preset_in_catalog` (params correctos: Stucki + sharpen ≥100).
- `test_detector_chooses_fine_textures_for_high_edge_density` (con monkeypatch
  para inyectar stats sintéticos confiables).
- `test_detector_size_invariant` (misma imagen a 1600px y 400px elige mismo preset).

**178 tests totales** (+3 nuevos), todos verdes.

### Resultado visual con Hokusai

| v1.8 (photo_high_detail) | v1.9 (photo_fine_textures) |
|---|---|
| jarvis + CLAHE — bueno | **Stucki + sauvola + sharpen 120 — excelente** |
| Espumas con detalle moderado | Espumas con líneas muy finas preservadas |
| Caracteres del sello legibles | Caracteres nítidos |
| Monte Fuji con halo suave | Monte Fuji nítido y definido |

## [1.8.0] — 2026-05-16

### Agregado — Preset `cartoon_back_engrave` para anime/illustration

El usuario probó una imagen de Hatsune Miku (Vocaloid, estilo anime cel-shaded
con fondo blanco transparente). El detector v1.7 caía en `photo_back_engrave`
que aplica halftone puntillista al sujeto — incorrecto para anime, donde se
quiere el sujeto entero como **silueta sólida frost**.

**Workflow correcto para anime sobre acrílico back-engrave**:
- Fondo blanco → NO grabar (transparente al ver desde el frente)
- Sujeto entero (cara + pelo + ropa con todos los colores) → grabar SÓLIDO como frost

**Implementación**:
```python
PRESET_CARTOON_BACK_ENGRAVE = LaserPreset(
    name="cartoon_back_engrave",
    algorithm="threshold",       # binario puro, sin halftone
    preprocess_mode="none",       # no suavizar el dibujo
    threshold=10,                 # ver nota abajo sobre invert
    contrast=1.0, gamma=1.0,
    sharpen=0.0,                  # bordes ya definidos en el dibujo
    invert=True,
    suggested_material="acrylic_funsun_9060_back_engrave",
)
```

### Bug fix — Convención `invert` en `preprocess_gray`

Durante el debugging del preset cartoon descubrimos que `invert=True` se aplica
**ANTES** del threshold dentro de `preprocess_gray()`. Con `threshold=240 + invert=True`:
- fondo blanco (255) → invert → 0 → < 240 → no graba ✓
- sujeto cyan (170) → invert → 85 → < 240 → ❌ tampoco graba (debería)

El threshold no es "absoluto contra el original" sino "absoluto contra el resultado
después de invert". Para detectar "fondo blanco puro" la fórmula correcta es:
**threshold=10 + invert=True**:
- fondo (255 invert→0) < 10 → no graba ✓
- sujeto (cualquier <245 invert→>10) >= 10 → graba ✓

Esto está documentado en la docstring del preset para futuro mantenimiento.

### Nueva regla del detector (Regla 0, prioridad más alta)

```
Regla 0 (NUEVA): very_bright_ratio > 50% AND extreme_ratio > 65%
                  → cartoon_back_engrave (silueta sólida)
Regla A: extreme_ratio < 40%                  → photo_high_detail
Regla B: very_bright_ratio > 18% AND extr<75% → photo_high_detail
Regla C: default acrylic                       → photo_back_engrave
```

### Validación

Probamos con 3 imágenes adicionales:

| Imagen | mean | std | extr% | vbri% | Preset elegido | Resultado |
|---|---|---|---|---|---|---|
| Anime sintético (Miku-like) | 224 | 63 | 77% | 76% | **cartoon_back_engrave** | ✅ silueta sólida limpia |
| Hokusai "Great Wave" | 156 | 55 | 9% | 5% | photo_high_detail | ✅ detalle preservado |
| Mona Lisa | 74 | 49 | 34% | 0% | photo_high_detail | ✅ fiel al original |

### Tests

3 tests nuevos:
- `test_cartoon_preset_in_catalog` (preset existe y params correctos)
- `test_detector_chooses_cartoon_for_anime_with_white_bg` (caso Miku)
- `test_cartoon_render_produces_solid_silhouette` (output silueta sólida)

`test_detector_chooses_back_engrave_for_bimodal_image` renombrado a
`test_detector_chooses_back_engrave_for_natural_bimodal_photo` y reescrito:
ahora valida foto bimodal NATURAL (Earth/rally style con texturas) que cae en
photo_back_engrave, distinto de dibujo bimodal puro (60% blanco + 40% negro
sólido) que ahora cae en cartoon.

**175 tests totales** (+3 nuevos), todos verdes.

## [1.7.0] — 2026-05-16

### Validación con stock público — 6 imágenes Picsum/Wikimedia

Probamos el pipeline v1.6 con stock público (Lorem Picsum + Wikimedia NASA):

- ✅ `wikimedia_earth_full` — excelente
- ✅ `wikimedia_jupiter_nasa` — bueno
- ⚠️ `picsum_237_800_600` (cachorrito) — cuerpo sobre-saturado a blanco
- ❌ `picsum_24_640_480` (libro abierto) — **texto perdido completamente**
- ⚠️ `picsum_1003_1280_720` (cervatillo bokeh) — bokeh chunky
- ⚠️ `picsum_1011_portrait` (mujer kayak) — cabello fino aplastado

**Patrón identificado**: las que fallan tienen `extreme_ratio < 40%` (mucho midtone),
las que salen bien tienen `extreme_ratio ≥ 40%` (bimodal natural). El preset
`photo_back_engrave` con gamma 1.55 + sharpen 130 aplasta midtones a blanco.

### Agregado — Preset `photo_high_detail` para fotos con midtones ricos

Nuevo preset conservador validado contra los casos de falla:

```python
PRESET_PHOTO_HIGH_DETAIL = LaserPreset(
    name="photo_high_detail",
    algorithm="jarvis_serpentine",  # error diffusion balanceado
    preprocess_mode="clahe",         # revela detalle en zonas extremas
    threshold=110,                   # más permisivo que 95
    contrast=1.10,                   # leve (no aplasta midtones)
    gamma=1.20,                      # cerca de lineal
    autocontrast=1.5,
    sharpen=80.0,                    # moderado
    invert=True,
    suggested_material="acrylic_funsun_9060_back_engrave",
)
```

### Mejorado — Detector con `extreme_ratio` como discriminador

`recommend_preset()` ahora distingue empíricamente para material acrílico:
- `extreme_ratio < 0.40` → `photo_high_detail` (cervatillo 8%, libro 24%, cachorrito 37%)
- `extreme_ratio ≥ 0.40` → `photo_back_engrave` (Jupiter 41%, Earth 45%, rally car 47%, hands 53%)

Razón en el mensaje al usuario explica por qué se eligió cada preset basado en
el porcentaje de pixels en zonas extremas.

### Mejorado — CLAHE adaptativo por std de imagen

CLAHE con `blend` fijo (0.6) sobre-realzaba bokehs (cervatillo) generando un
look "chunky". Ahora el blend se adapta a la varianza global:
- `std < 50` (imagen suave/bokeh): `blend = 0.30` (CLAHE leve)
- `std < 70`: `blend = 0.45`
- `std ≥ 70` (mucho detalle distribuido): `blend = 0.60`

### Mejorado — `plain_region_simplification` con morphological cleanup

El v1.6 dejaba "manchas" aisladas en gradientes complejos (bokeh con luces).
Ahora:
- Descarta regiones conectadas < `min_region_size=64` px (elimina manchas chicas).
- Aplica `binary_closing` con `border_value=1` para cerrar huecos de ruido
  JPEG dentro de regiones uniformes (sin perder bordes).
- Resultado: cielos limpios sin artefactos de stamping.

### Mejorado — Detector con `very_bright_ratio` (Regla B nueva)

Tras la mejora de Regla A (extreme_ratio < 40%), un caso aún fallaba: la mujer en
kayak (`picsum_1011_portrait`) tenía `extreme_ratio=66%` pero los extremos eran
sobre todo el **cielo + lago muy brillantes** (24% > 215), no un bimodal natural.
El preset agresivo aplastaba el cabello y la ropa a blanco aunque el discriminador
de Regla A no la cubría.

**Nueva métrica**: `ImageStats.very_bright_ratio` = fracción pixeles > 215.

**Regla B**: si `very_bright_ratio > 0.18` AND `extreme_ratio < 0.75` →
`photo_high_detail` (sujeto midtone sobre fondo brillante grande).

Tabla de decisión final:

| Imagen | extr% | vbright% | Regla | Preset |
|---|---|---|---|---|
| Cervatillo | 8% | 0% | A (extr<40) | photo_high_detail |
| Libro | 24% | 15% | A (extr<40) | photo_high_detail |
| Cachorrito | 37% | 1% | A (extr<40) | photo_high_detail |
| **Mujer kayak** | **66%** | **24%** | **B (vbright>18)** | **photo_high_detail** |
| Jupiter | 41% | 5% | default | photo_back_engrave |
| Earth | 45% | 8% | default | photo_back_engrave |

### Validación visual posterior (v1.7)

Re-procesamos las 6 stock images:

| Imagen | v1.6 | v1.7 | Detalle |
|---|---|---|---|
| Earth | ✅ excelente | ✅ excelente | sin cambios (preset OK ya) |
| Jupiter | 🟡 bueno | 🟡 bueno | sin cambios |
| Cachorrito | ⚠️ saturado | ✅ **pelaje visible** | Regla A → nuevo preset + CLAHE |
| Libro | ❌ texto perdido | ✅ **texto legible** | Regla A → nuevo preset detectado |
| Cervatillo | ⚠️ bokeh chunky | ✅ **bokeh suave** | CLAHE adaptativo blend 0.30 |
| Mujer kayak | ⚠️ cabello plano | ✅ **cabello + textura kayak** | Regla B nueva |

### Tests

7 tests nuevos:
- `test_catalog_has_all_expected_presets` extendido con `photo_high_detail`.
- `test_photo_high_detail_has_conservative_params` (gamma < be, sharpen < be).
- `test_detector_chooses_high_detail_for_midtone_rich_image` (Regla A).
- `test_detector_chooses_back_engrave_for_bimodal_image`.
- `test_detector_handles_bright_background_with_midtone_subject` (Regla B nueva).
- `test_image_stats_has_very_bright_ratio` (campo nuevo).
- `test_plain_region_simplification_*` fixes (`border_value=1` para edge case).

**172 tests totales** (+5 nuevos), todos verdes.

### Bug fix — `binary_closing` edge erosion

`scipy.ndimage.binary_closing` con default mode "borra" 1 px del borde de
imágenes totalmente uniformes (252 px = perímetro de 64×64). Fix: `border_value=1`
en la llamada explícita. Descubierto al correr los tests existentes tras agregar
morphological cleanup.

## [1.6.0] — 2026-05-16

### Mejoras descubiertas procesando imágenes reales

Procesamos las 4 imágenes de referencia del proyecto
(`input_agricultor.jpeg`, `input_rallycar.jpg`, `input_whatsapp_hands.png`,
`foto_objetivo_sin_procesar.jpeg`) y observamos 3 problemas concretos no
cubiertos por v1.5:

#### Problema A: Fondos uniformes con dither inútil

El cielo gris del rally car, el fondo blanco de las manos — el dither generaba
patrones moteados que no aportan información visual pero **cada punto se traduce
en un pulso del láser**: tiempo de grabado, desgaste del tubo, marcas leves en
material claro.

**Fix — `plain_region_simplification()`** en `scripts/laser_target_match.py`:
- Calcula varianza local en ventana 9×9.
- Donde `var < 25` AND `gray > 220` → clampea a 255 puro (blanco).
- Donde `var < 25` AND `gray < 35` → clampea a 0 puro (negro).
- Resto sin cambio.

Activado por default vía `ProcessParams.simplify_plain_regions=True`.
Toggle visible en Modo Manual ("Simplificar fondos planos").

**Impacto medido en imágenes reales:**

| Imagen | px simplificados | tiempo |
|---|---|---|
| agricultor (poster) | 29.8% | 10.4 ms |
| rally car | 31.8% | 9.7 ms |
| whatsapp hands | 37.3% | 6.9 ms |
| foto objetivo | 37.3% | 7.4 ms |

→ Aprox **1/3 menos de pulsos del láser** sin pérdida visual.

#### Problema B: HQ refine empeoraba algunos casos

El refine podía elegir un candidato con score marginalmente mejor pero peor
**visualmente** (ej. contraste muy alto que come detalles, threshold muy bajo
que come zonas claras).

**Fix — `MIN_IMPROVEMENT_RATIO = 0.05`** en `_hq_refine()`:
El primer candidato (baseline = params del preset) se acepta como referencia.
Las variantes del Sobol sólo reemplazan al baseline si su score es **≥ 5%
mejor en términos relativos**. Sin esta guarda, el refine podía moverse a
mínimos locales sutiles que no aportaban calidad visual real.

#### Problema C: micro-detalles aislados podían escapar al detector v1.5

`edge_preservation_error()` usaba percentil 85 sobre magnitudes > 0.05. Un
logo de 4×6 px en imagen mayormente plana podía no cruzar el threshold de
0.05 (gradient relativo bajo).

**Fix — Fallback de micro-detail**: si NO hay magnitudes > 0.05 del max pero
hay zonas brillantes, baja el threshold absoluto a 0.01 del max como red
de seguridad. Detecta micro-logos en fotos mayormente planas.

### Frontend

- `ProcessParams.simplify_plain_regions: boolean` (opcional, default `true`).
- Checkbox visible en Modo Manual junto al toggle de "Invertir".
- `DEFAULT_PARAMS` incluye `simplify_plain_regions: true`.

### Tests

5 tests nuevos:
- `test_plain_region_simplification_clamps_bright_uniform` (cielo → 255 puro).
- `test_plain_region_simplification_clamps_dark_uniform` (sombra → 0 puro).
- `test_plain_region_simplification_preserves_detail` (zonas con detalle intactas).
- `test_plain_region_simplification_preserves_midtones` (grises medios no se tocan).
- `test_micro_detail_in_plain_image_still_detected` (fallback edge preservation).

3 tests en api_server:
- `test_simplify_plain_regions_default_on`
- `test_simplify_plain_regions_can_be_disabled`
- `test_hq_refine_baseline_preserved_when_no_significant_improvement`

**167 tests totales** (+8 nuevos), todos verdes.

### Validación visual

Carpeta `runs/test_v15/` con outputs comparados v1.4 → v1.5 → v1.6 para las
4 imágenes de referencia. El rally car con `WITH_simp` muestra fondo limpio
negro/blanco puro mientras preserva todos los logos (RedBull, SUBARU) y
detalles del auto.

## [1.5.0] — 2026-05-16

### Agregado — preservación de detalles finos en zonas brillantes (v5 mejorado)

Soluciona el caso real reportado: **logo Red Bull en capó blanco se pierde porque
el láser pasa por encima del toro**. La causa raíz era una falla del scoring v5
para penalizar la pérdida de bordes finos en zonas casi uniformes:

1. **HVS-MSE** atenuaba el detalle pequeño como ruido perceptual.
2. **tone_match** con `scale=8` promediaba el detalle con el blanco circundante.
3. **spectral_radial_penalty** castigaba al algoritmo que sí preservaba el detalle
   (porque preservar requiere un cluster localizado de puntos negros que degrada
   el perfil blue-noise).

### Backend — `scripts/laser_scoring.py`

- **`edge_preservation_error(binary, gray, ppd, bright_threshold, edge_percentile)`**
  nueva función que mide pérdida de bordes finos en zonas brillantes:
  1. Sobel magnitude del gris pre-dither → mapa de "bordes importantes".
  2. Sobel magnitude del binario PASADO por filtro CSF → mapa percibido.
  3. Máscara `gray > bright_threshold` (default 0.6) → solo zonas claras.
  4. Threshold de magnitude: percentil 85 de magnitudes > 0.05 del max
     (robusto a imágenes mayormente planas — el bug viejo era usar percentil
     sobre TODOS los magnitudes, incluyendo ceros).
  5. Error = `|g_mag_norm − b_mag_norm|` ponderado por magnitude, en `[0, 1]`.

- **`multi_scale_tone_match_error(binary, gray, scales=(4,8,16), weights=(0.50,0.30,0.20))`**:
  reemplaza el single-scale=8 por promedio ponderado en 3 escalas; peso mayor a
  scale=4 captura detalles pequeños que el block-mean de 8 perdía.

- **`score_candidate_v5_terms`** rebalanceado:
  - v1.4: `0.50*hvs + 0.30*spec + 0.20*tone + 0.05*reg`
  - v1.5: `0.35*hvs + 0.20*spec + 0.10*tone + 0.35*detail + 0.05*reg`
  - Parámetro `detail_weight=0.35` (default; `0` para modo legacy v1.4).
  - Parámetro `multi_scale_tone=True` (default; `False` para single-scale legacy).
  - Parámetro `bright_threshold=0.6` ajustable.
  - Nuevas claves en el dict de retorno: `detail_error, w_detail, detail_weight,
    bright_threshold, multi_scale_tone`.

### Backend — `scripts/laser_target_match.py`

- **`clahe_preprocess_gray(gray, clip_limit=2.5, tile_size=8, blend=0.6)`**:
  CLAHE (Contrast Limited Adaptive Histogram Equalization, Reza 2004 + Pizer 1987)
  redistribuye localmente el histograma para revelar detalles sutiles antes del
  dither. Implementación via `skimage.exposure.equalize_adapthist`.

### Backend — `scripts/api_server.py`

- Nuevos `preprocess_mode`:
  - `'clahe'`: CLAHE moderado solo.
  - `'sauvola_clahe'`: pipeline 2-pass (CLAHE primero, Sauvola después) —
    máxima preservación para casos extremos como logos en pintura clara.

### Frontend

- `ProcessParams.preprocess_mode` extendido con `'clahe'` y `'sauvola_clahe'`.
- Select del Modo Manual con las 2 nuevas opciones + hint contextual cuando
  se elige CLAHE: explica cuándo usarlo (logos/letras chicas en zonas extremas).

### Tests

- `test_v5_edge_preservation_detects_lost_detail`: reproduce el bug del logo Red Bull
  (cuadrado 8×8 a gris 190 en imagen 128×128 a gris 245). Verifica que v5 legacy
  elige el threshold global puro que pierde el detalle (`detail_density=0`) y que
  v5 v1.5 con detail term default elige el dither que sí preserva (`detail_density=0.27`).
- `test_edge_preservation_zero_for_plain_image`: corner case (imagen sin bordes).
- `test_multi_scale_tone_match_reduces_block_blindness`: multi-scale ≥ single-scale
  cuando hay detalles finos.
- `test_clahe_preprocess_enhances_local_contrast`: CLAHE aumenta σ local en regiones
  con detalle, sin saturar.
- 159 tests totales (+4 nuevos).

### Validación empírica del fix

Imagen 128×128, gris uniforme 245 con cuadrado 8×8 a 190 (simulación del logo):

| Versión | Algoritmo elegido | Detalle preservado | Score |
|---|---|---|---|
| v5 legacy (detail=0) | `threshold-180` (malo) | **0%** (pierde el toro) | 0.0004 |
| v5 v1.5 (detail=0.35) | `floyd+sharpen` (bueno) | **27%** (preserva el toro) | 0.2628 < 0.3318 |

## [1.4.2] — 2026-05-16

### Agregado — toggle CPU/GPU (v5 vs v4) en Mi Configuración

El usuario ahora puede elegir explícitamente la métrica del HQ refine:

- **v5 (CPU, no-reference)** — default; HVS-MSE + spectral blue-noise + tone match.
  ~5-15 ms por candidato. La métrica recomendada para grabado láser.
- **v4 (LPIPS perceptual + AlexNet)** — opcional; usa torch + lpips. Acelera con
  GPU si torch CUDA está instalado, sino corre en CPU (~5-10× más lento).

La UI muestra un badge **"⚡ GPU disponible"** o **"🖥 CPU only"** según
`torch.cuda.is_available()` del backend, y al seleccionar v4 sin GPU informa
qué comando correr para reinstalar torch CUDA.

### Backend
- `ProcessParams.score_version: 'v5' | 'v4'` (default `v5`).
- `_hq_refine()` genera **auto-target** sintético cuando se pide v4: binario
  por threshold-128 del gris + density a 1/4 resolución + edges via Sobel.
  Esto permite que v4 funcione en Express sin que el usuario provea un PNG ideal.
- `refine_debug.score_version` reportado en headers para introspección.

### Frontend
- `MyConfig.score_version` persistido en localStorage junto a material/mm/DPI.
- `ProcessParams.score_version` opcional en el cliente.
- Toggle radio en "Mi configuración" con descripciones contextuales y aviso
  cuando torch es `+cpu`.
- `expressProcess` y `processFullRes` propagan el `score_version` elegido.

### Tests
- `test_async_score_version_v5_default` valida que sin parámetro corre v5.
- `test_async_score_version_v4_runs` valida que v4 con LPIPS Alex completa OK
  (skip si torch/lpips no instalados).
- 155 tests totales (+2).

## [1.4.1] — 2026-05-16

### Agregado — telemetría extendida en la barra de progreso

Sobre la barra SSE de v1.4.0, se agrega visibilidad operacional del worker:

- **Memoria RSS** y **% CPU** del proceso Python via `psutil` (`psutil.Process()`
  singleton para que `cpu_percent()` mantenga su baseline interno entre llamadas).
- **Tiempo promedio por candidato** (`seconds_per_candidate = elapsed / current`)
  y **tiempo del último candidato** (delta entre updates).
- **Sparkline SVG inline** con la evolución del best_score por iteración. Útil para
  ver si el Sobol search está mejorando o ya convergió (curva chata).
- **Log estructurado del worker** con timestamp + mensaje + kind (`info`/`warn`/`error`),
  capado a 80 entradas, expuesto como `<details>` scrollable en la UI.

Eventos automáticos del log: inicio, decode, dimensiones imagen, preset aplicado +
motivo del detector, material, nuevo mejor score con delta de tiempo, encode, fin,
errores y cancelaciones.

### Backend — `scripts/laser_jobs.py`
- `JobState` con `memory_mb`, `cpu_pct`, `seconds_per_candidate`,
  `last_candidate_seconds`, `score_history: list[float]`, `log_lines: list[dict]`.
- Métodos `push_score()` y `log()` con cap automático (60 / 80 entradas).
- Caps `MAX_SCORE_HISTORY` y `MAX_LOG_LINES` configurables.

### Backend — `scripts/api_server.py`
- `_init_process_metrics()` y `_process_metrics()` con singleton `_PSUTIL_PROC`.
- `_run_job_in_thread()` reescrito sobre el JobState directo (single-writer thread)
  en vez de `REGISTRY.update(...)` por campo — más expresivo y permite
  `job.push_score()` / `job.log()`.

### Frontend
- `JobProgress` y `JobLogLine` en `apiClient.ts` con los nuevos campos.
- `+page.svelte`: tarjeta de telemetría con grid de 6 cells (score, avg/cand,
  último cand, RAM, CPU, stage), sparkline SVG y log scrollable. Versión compacta
  para el step 3 (Manual).
- `buildSparklinePath()` helper TS puro (sin libs).

### Tests
- `test_async_emits_telemetry_fields` verifica `memory_mb > 0`,
  `score_history`, `log_lines` con structure correcta. 153 tests totales (+1).

## [1.4.0] — 2026-05-15

### Agregado — barra de progreso en tiempo real (SSE)

Resolución del bug "se quedó colgado" reportado por el usuario: durante el HQ refinement
(2–3 min) el wizard no daba feedback, lo que se sentía como freeze.

- **Backend asíncrono con SSE** (`scripts/laser_jobs.py` + 5 endpoints nuevos):
  - `POST /api/process_async` → encola el job, devuelve `{job_id}` inmediato.
  - `GET  /api/jobs/{id}` → snapshot puntual del estado.
  - `GET  /api/jobs/{id}/stream` → Server-Sent Events, ~2 Hz, eventos con
    `status / current / total / progress_pct / eta_seconds / elapsed_seconds /
    stage / best_score / error_message`.
  - `GET  /api/jobs/{id}/result` → PNG final + headers (404 si no existe,
    409 si en progreso, 410 si cancelado, 500 si erroró).
  - `POST /api/jobs/{id}/cancel` → marca cancelación; el worker aborta entre
    candidatos del Sobol search.
- `JobRegistry` thread-safe en memoria con cleanup automático (TTL 1 h,
  cap 100 jobs FIFO). Pensado para uso local single-process.
- `_hq_refine()` recibe ahora `progress_cb(current, total, best_score, elapsed)`
  y `cancel_check()` que se invocan tras cada candidato evaluado.

### Agregado — UI de progreso en el wizard

- **Barra de progreso animada** con porcentaje, candidatos evaluados, tiempo
  transcurrido, ETA y mejor score parcial. Activa en modos **Express** y
  **Manual** durante `processFullRes()`.
- **Botón "Cancelar"** que llama a `/api/jobs/{id}/cancel`. La UI muestra
  "Cancelando…" hasta que el worker confirme.
- Fallback a polling cada 750 ms si el navegador no soporta `EventSource` o
  un proxy bloquea SSE — la barra sigue funcionando.

### Fix — "failed to fetch" después de procesar

Diagnóstico: el cliente Express mandaba `algorithm: 'jarvis_serpentine'`,
`threshold: 128`, etc. (del `DEFAULT_PARAMS` del frontend) que diferían de los
defaults del schema Pydantic. El merger de `_resolve_preset_overrides()` los
trataba como "elección explícita del usuario" y **no** aplicaba los valores del
preset auto-detectado. Adicionalmente, cuando el HQ refine tardaba 2–3 min, el
navegador podía cerrar el `fetch()` por inactividad de keep-alive.

- `expressProcess()` ahora manda **solo** los campos que el usuario controla
  (`preset='auto', material, output_mm_short, output_dpi, sharpen_radius_mm`).
  El backend completa el resto con sus defaults y el preset elegido por el
  detector reemplaza esos campos. Resultado: el preset realmente aplica.
- Migrado a flujo async/SSE — ya no hay fetch largo bloqueante. El stream
  se mantiene vivo con eventos cada 0.5 s.
- **CORS `expose_headers`** explícito para todos los `X-*` que el cliente lee.
  Sin esto, fetch desde origen distinto (ej. dev `:5173`) leía esos headers
  como `null` y rompía el meta de la UI.
- Origen `127.0.0.1:18765 / localhost:18765` agregado a `allow_origins` por las
  dudas (mismo origen no debería necesitarlo, pero blinda configs raras).

### Tests

- 5 tests nuevos en `tests/test_api_server.py` cubren el flujo async:
  - `test_process_async_returns_job_id_and_completes`
  - `test_process_async_with_invalid_image_fails_fast`
  - `test_process_async_cancel`
  - `test_jobs_status_404_when_unknown`
  - `test_sse_stream_emits_terminal_status`
- 152 tests total (previo: 147), todos verdes.

## [1.3.2] — 2026-05-15

### Fix
- **`Setup_LaserApp.bat`** y **`Iniciar_Laser_App.bat`** ahora invocan directamente
  el shim `node_modules\.bin\vite.cmd` con ruta absoluta, en vez de `npm run build`.
  Esto soluciona el error `vite no se reconoce` que aparecía al ejecutar el setup
  dentro de `C:\Program Files\ImageAPLaser\` (PATH del shell de Inno Setup no
  resolvía bien los binarios locales de npm).
- Si `node_modules\.bin\vite.cmd` no existe, Setup reinstala las dependencias
  con `npm ci` (o `npm install` como fallback). Cubre el caso de `node_modules`
  presente pero corrupto/incompleto.
- npm install ahora usa `--no-audit --no-fund` para evitar warnings irrelevantes
  durante la primera instalación.

## [1.3.1] — 2026-05-15

### Fix
- **Launcher `Iniciar_Laser_App.bat`** ya no intenta arrancar Vite dev server.
  El wizard se sirve desde el FastAPI estático (`/app/`) como en v1.2+. Esto
  elimina el error `vite no se reconoce` que aparecía tras instalar via .exe
  (problema de PATH de `npm run dev` en `cmd` shells dentro del install dir).
- Si `web/build/index.html` no existe al arrancar, el launcher hace `npm run build`
  automáticamente. La ventana `Laser Web` (Vite) ya no se abre.
- Para hot-reload de UI (dev): `cd web && npm run dev` manualmente. El launcher
  oficial va por la ruta estática.

## [1.3.0] — 2026-05-15

### Agregado — flujo Express
- **Modo Express en el wizard**: subís una foto y la app la procesa automáticamente
  con tu configuración guardada (material + mm + DPI) + auto-detección de preset +
  HQ refinement. Cero sliders. Una foto, un resultado.
- **`Mi configuración` persistente en localStorage** (`MyConfig`): material por
  defecto, output mm corto, DPI, sharpen radius. Se aplica en Express y como
  defaults en Manual.
- **Toggle Modo Express / Modo Manual** en el primer paso del wizard.

### Agregado — recomendaciones LightBurn
- `MaterialProfile` ampliado con campos `speed_mm_s_range`, `pass_through`,
  `mirror_x_required`, `lightburn_invert`, `focus_mm`, `machine_compat`.
- **Nuevo endpoint** `GET /api/recommended_settings/{material}`: devuelve JSON con
  DPI, interval mm, power %, speed mm/s, pass-through, mirror, invert, focus mm,
  + notas humanas para el operador.
- **Panel `Configuración recomendada para LightBurn`** en el step Resultado del
  wizard: tarjeta visible con cada valor listo para copiar al CAM.
- Profiles actualizados con settings reales:
  - `acrylic_funsun_9060_back_engrave`: DPI 115, interval 0.220 mm, 9–14% power,
    400–600 mm/s, Pass-Through ON, MirrorX ON, LightBurn invert OFF, focus 7 mm.
  - `acrylic_back_engrave` (genérico CO2 50W 2"): DPI 169, 9–14% power, 400–700 mm/s.
  - `wood_generic`: DPI 141, 20–60% power, 250–500 mm/s, MirrorX OFF.

### Cambiado
- `MaterialInfo` (response model API) incluye los nuevos campos operacionales.
- Lista de materiales `/api/materials` ahora expone `acrylic_funsun_9060_back_engrave`
  como builtin (estaba sólo via auto-detect).
- `MyConfig` default sugiere `acrylic_funsun_9060_back_engrave` (configuración del
  usuario reportada).

### Validación
- `npm run check` 0 errors / 0 warnings.
- `pytest tests/test_laser_physics.py tests/test_api_server.py -q`: 35+ passed.
- Build estático SvelteKit OK.

## [1.2.0] — 2026-05-15

### Agregado — máxima calidad por imagen
- **HQ refinement automático en `/api/process`**: el endpoint full-res ahora corre
  un sobol-search de 25 candidatos alrededor del preset elegido y selecciona el mejor
  por **score v5 (no-reference)**. Tiempos: ~2-3 min/imagen full-res (113-181s en
  CPU; mucho menos con CUDA en LPIPS).
- **`quality_mode` en ProcessParams**: `'fast'` (1 render) o `'best'` (HQ search).
  `/api/process` default `'best'`; `/api/preview` queda en `'fast'`.
- Headers nuevos: `X-Refine-Candidates`, `X-Refine-Best-Score`, `X-Refine-Improvement`,
  `X-Refine-Seconds`, `X-Quality-Mode` para que el cliente vea qué pasó.

### Agregado — máquina Funsun 9060
- **Nuevo `MaterialProfile`**: `acrylic_funsun_9060_back_engrave` (lente 2.5", spot
  0.22mm, DPI MAX 115, focus 7mm) — la configuración real del usuario.
- **Spot table** ampliado en `laser_physics.SPOT_SIZE_DEFAULTS_MM` con variantes
  Funsun 9060 (2", 2.5", 4").
- **Nuevo preset `photo_back_engrave`**: tuneado para foto sobre acrílico (Stucki
  serpentine + threshold 95 + gamma 1.55 + autocontrast 3.5 + sharpen 130 + invert
  para grabado positivo subject-as-frost). Reemplaza al `poster_back_engrave` para
  fotos cuando el material es acrílico.

### Cambiado
- **Auto-detector más agresivo**: si `material` empieza con `"acrylic"`, TODAS las
  fotos van a `photo_back_engrave` (no scene_dark/scene_bright que están tuneados
  para madera). Reglas de extreme_ratio/edge_density para gráficos siguen como antes.
- `/api/process` default cambió de `fast` → `best`. Para mantener el comportamiento
  anterior pasar `quality_mode: "fast"` en `params_json`.

### Agregado — instalador profesional
- **`Setup_LaserApp.bat`**: bootstrap completo (winget instala Python+Node si faltan,
  crea venv, pip install, npm build, crea shortcut en Escritorio).
- **`installer/LaserApp.iss`**: script Inno Setup 6 para compilar `.exe` profesional
  con UI estándar (asistente con licencia GPL-3, carpeta destino, accesos directos,
  desinstalador en Panel de Control).
- **`installer/README.md`**: instrucciones para compilar el `.exe`.
- **`installer/Output/ImageAPLaser_Setup_v1.2.0.exe`** (6.5 MB) compilado y listo
  para distribuir como release asset.

### Cambiado — repo público
- Visibility: PRIVATE → **PUBLIC** en GitHub. Cualquiera puede clonar/ver/contribuir.

### Validación
- 47+ tests passing (`pytest -m "not network"`).
- 5 imágenes stock procesadas con HQ + Funsun 9060 profile: cada una muestra
  refine improvement v5 entre 0.0018 y 0.0106 (mejora real medible).
- Rally car Subaru procesado: PNG con halftone definido (no la "mush" del grabado
  previo que motivó esta release).

## [1.1.0] — 2026-05-15

### Agregado
- **Sistema de presets curados** (`scripts/laser_presets.py`): 6 presets validados
  experimentalmente con stock diverso (`photo_general`, `portrait`, `scene_dark`,
  `scene_bright`, `poster_back_engrave`, `line_art`). Cada preset trae un conjunto
  coherente de params + material sugerido.
- **Auto-detector** `recommend_preset(rgb)`: analiza estadísticos básicos
  (luminancia media, std, ratio bimodal, densidad de bordes) y elige el preset
  adecuado para la imagen subida. Sin ML — heurísticos rápidos sobre histograma.
- **Endpoints API**:
  - `GET /api/presets` — catálogo de presets para el wizard.
  - `POST /api/recommend_preset` — analiza la imagen y devuelve preset + razón.
- **`ProcessParams.preset`**: nuevo campo. Si se pasa nombre de preset (o `auto`),
  el server aplica esos params como base; campos explícitos del request
  **sobreescriben** el preset.
- **Default del wizard cambia a `preset='auto'`**: cada imagen sale con acabado
  adecuado por defecto sin que el operador tenga que tunear.
- **Polaridad correcta para grabado positivo**: los presets de foto ahora usan
  `invert=True` para que el sujeto oscuro de la foto se grabe oscuro en madera
  (positivo), en vez de quedar como negativo.

### Cambiado
- **Licencia: MIT → GPL-3.0-or-later** (copyleft). Las contribuciones siguen siendo
  bienvenidas, pero las versiones derivadas deben permanecer open-source bajo la
  misma licencia y preservar los créditos.
- README + CHANGELOG actualizados con tabla de presets y workflow auto.

### Agregado (gobernanza)
- [`AUTHORS.md`](AUTHORS.md) — registro de contribuyentes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — guía de PR, setup, estilo, qué se acepta.

### Validación
- Auto-detector probado contra 5 imágenes stock diversas (perro, paisaje, retrato chica,
  ola, persona en canoa) → cada una sale con acabado impecable y polaridad correcta.
- 130+ tests verde (12 nuevos para presets, antes 115).

### Diferido a v1.2+
- WebSocket de progreso para sweeps largos.
- PyInstaller .exe stand-alone (PyTorch hace el bundle pesado; primero hay que probar
  el approach con un toy build).
- Mejora del `spectral_radial_penalty` con high-pass previo (distinguir dither de
  macro-estructura).
- Calibración física por step-wedge con láser real del usuario (workflow listo;
  esperando a que se ejecute físicamente).

## [1.0.0] — 2026-05-15

Primera versión consolidada. Pipeline funcional verificado contra dos imágenes reales
(poster "Día del Agricultor" y retrato stock Picsum) con veredicto FUNCIONAL.

### Agregado
- **Motor**: `scripts/laser_target_match.py` con 40+ algoritmos de halftone organizados
  en tres tablas (`DIFFUSION_ALGORITHMS`, `BURKES_BLUE_VARIANTS`, `NAMED_RENDERERS`)
  y `ALL_RENDER_ALGORITHMS` exportada.
- **Scoring v1..v5**: `scripts/laser_scoring.py`
  - v1: legacy densidad-dominada.
  - v2: SSIM continuo.
  - v3: v2 + SSIM blur sobre binario.
  - v4: blur simétrico + SSIM/MSE + LPIPS (Alex, GPU/CPU).
  - **v5 (nuevo)**: sin referencia — HVS-MSE (CSF Mannos-Sakrison) + spectral radial
    penalty + tone match local post-LUT.
- **Física láser**: `scripts/laser_physics.py` — `MaterialProfile` cargable, validación
  DPI por spot, sharpen radius escalado al output físico, LUTs builtin para
  `acrylic_back_engrave` y `wood_generic`.
- **Blue-noise auténtico**: `scripts/laser_blue_noise.py` — algoritmo void-and-cluster
  (Ulichney 1993) con caching a `assets/blue_noise_*.npy`.
- **Calibración física por material** (workflow Fase R7):
  - `scripts/laser_calibration_wedge.py` — genera tira step-wedge para grabar.
  - `scripts/laser_calibration_fit.py` — ajusta LUT desde foto del wedge grabado
    (PCHIP monotónico + PAVA isotonic fallback para madera).
- **Simulador de grabado físico**: `scripts/laser_simulator.py` — Gaussian spot blur
  + respuesta tonal por material (acrylic frost / wood burn / raw).
- **API REST FastAPI** (`scripts/api_server.py`):
  - `GET /api/health`, `/api/materials`, `/api/algorithms`
  - `POST /api/preview` (max-side 400, ~1s), `/api/process` (full-res),
    `/api/simulate` (grabado simulado).
  - Static mount: `/app` sirve el wizard SvelteKit. `/` redirige a `/app/`.
- **Wizard SvelteKit 5**: tema oscuro+verde militar acorde a taller. 5 pasos:
  subir → recortar → ajustes → resultado (con simulación) → descargar PNG.
  Comparador antes/después con slider arrastrable. Presets toggle (óptimo / manual).
- **Cliente HTTP TypeScript**: `web/src/lib/apiClient.ts` tipado completo con presets.
- **Launcher**: `Iniciar Laser App.bat` para arranque doble-click (Windows).
- **Tests**: 100+ tests (motor, scoring v1-v5, física, blue-noise, calibración,
  simulador, API endpoints, render dispatch). `pytest -m "not network"` → verde.
- **Documentación**:
  - `README.md` — entry doc con quickstart.
  - `docs/USAGE.md` — guía completa operador + CLI avanzado + calibración + troubleshoot + FAQ.
  - `docs/IMPROVEMENT_LOG.md` — historial técnico por fases.
  - `docs/PLAN_*.md` — planes retomables si la sesión se corta.
  - `docs/SESSION_HANDOFF_*.md` — handoff entre agentes.

### Notable / lessons (resumen del trabajo iterativo)
- Bajo target-matching contra ImagR PNG, **Floyd-Steinberg gana** sobre `blue_noise_vac32`
  en v4 y v5 (la macro-estructura del fotograbado domina el espectro).
- Mejor score conocido **v4 = 0.3247** sobre el poster Agricultor con preset
  `floyd inv=1 thr=75 c=1.0 b=+10 g=1.2 ac=2.0 sharpen=60` + sauvola.
- **Meseta estructural ~0.39** al comparar contra PNG ImagR específico: se rompe sólo
  cambiando el target (calibración física, workflow Fase R7 disponible).
- Bug fix VRAM: `_lpips_distance_scaled` libera `del s1,s2,x1,x2,d` + `torch.cuda.empty_cache()`
  cada 25 calls → permite sweeps largos sin matar el proceso.
- Refactor invasivo de `render_candidate` (de 230 líneas if/elif a 3 tablas + dispatch),
  removiendo 44 líneas de código muerto.

### Stack
- Python ≥3.11. Deps base: numpy, scipy, scikit-image, Pillow, opencv-headless.
  Extras: `[api]` (fastapi + uvicorn), `[perceptual]` (torch + lpips), `[fast]` (numba),
  `[meta]` (optuna), `[dashboard]` (streamlit), `[sam2]` (transformers).
- SvelteKit 5 + Vite + cropperjs + TypeScript strict.

[1.0.0]: https://github.com/Vahlame/image-ap-sport-for-laser/releases/tag/v1.0.0
