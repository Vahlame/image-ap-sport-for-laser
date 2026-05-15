# Changelog

Todas las versiones notables del proyecto se documentan acá.
Formato basado en [Keep a Changelog](https://keepachangelog.com/) +
[Semver](https://semver.org/).

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
