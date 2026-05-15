# Changelog

Todas las versiones notables del proyecto se documentan acá.
Formato basado en [Keep a Changelog](https://keepachangelog.com/) +
[Semver](https://semver.org/).

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
