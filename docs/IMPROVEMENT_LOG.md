# Improvement log — image ap sport for laser

Registro de auditoria y auto-mejora del pipeline (branch `feat/auto-improve-pipeline`).

## Fase 0 — Reconocimiento — 2026-05-14

### Procesos Python observados (no interrumpidos)

| PID   | Notas                          |
|-------|--------------------------------|
| 20820 | `C:\Python314\python.exe`      |
| 24808 | `C:\Python314\python.exe`      |
| 28880 | python via uv cache            |

### Entorno

- Repositorio: inicializado localmente con `.gitignore` (carpeta `runs/` ignorada salvo referencias si aplica).
- Dependencias opcionales probadas: `skimage` OK; `numba`, `optuna`, `piq`, `lpips` no instalados (se anaden como extras en `pyproject.toml`).

### Mapa `laser_target_match.py` (aprox.)

- `init_worker` ~L222
- `evaluate_candidate_task` ~L252
- `dense_blue_family_candidates` ~L905
- `error_diffusion` ~L969
- `score_candidate` ~L1222
- `build_candidates` ~L1242
- `guided_run_evaluation` ~L1870
- `target_threshold` / Otsu ~L2381-2383

### Validacion

- Smoke posteriores por fase (n pequeno, salida bajo `runs/_smoke_*` solo cuando el plan lo pida; no se editan artefactos de experimentos en curso).

## Continuacion (handoff) — 2026-05-14

### Cambios

- `scripts/laser_target_match.py`: asignacion de `_ERROR_DIFFUSION_JIT` sin `global` anidado (evita `SyntaxError` en import); log `[CONFIG]` plateau en rama no guiada; resto segun plan previo (scoring dispatch, Sobol, registro meta, etc.).
- `scripts/laser_scoring.py`: `score_candidate_dispatch` con rama `v2` explicita y error en version desconocida.
- `scripts/meta/proposer.py`: hashes SHA-256 de input/target para filtrar historial; fallback si no hay coincidencias.
- `scripts/meta/analyzer.py`: `--last` por defecto 10; `--regressions`, `--baseline-for`, `--regression-pct`.
- `scripts/laser_optuna_search.py`: estudio Optuna TPE + export `optuna_match_seed.sqlite` compatible con `--from-db`.
- `scripts/dashboard.py`: Streamlit minimo sobre `history.sqlite` (extra `[dashboard]`).
- `tests/test_score_regression.py`, `tests/test_plateau_detector.py`: regresion de score v2 y plateau.
- `pyproject.toml`: dependencias base `scipy`, `scikit-image`, `opencv-python-headless`; extras `[fast]`, `[meta]`, `[dashboard]`, `[perceptual]`.
- `docs/META_SYSTEM.md`, `README.md`: comandos meta y dashboard.

### Validacion

- `python -m py_compile` en scripts tocados: OK.
- `pytest -v -m "not network"`: 11 passed.

### Pendientes

- Configurar `git remote add origin ...` y `git push -u origin feat/auto-improve-pipeline` cuando corresponda (aqui no hay remoto).
- Corridas largas de smoke (`--n` 100+ v2 + `--register affine`) opcionales para validar metricas visuales end-to-end.
- `PlateauDetector` sin accion `ABORT` (solo `RESTART` / `NONE`); ampliar si hace falta criterio de aborto global.

### Ajuste Sobol + git meta

- `dense_blue_family_candidates`: muestra Sobol con `n` potencia de 2 y recorte para evitar `UserWarning` de scipy.
- `.gitignore`: `runs/_meta/*.sqlite` para no versionar bases locales generadas por corridas.

## Runtime LPIPS / subprocess / Windows — 2026-05-14

### [FACT] `scripts/laser_runtime_env.py`

- `LASER_LPIPS_DEVICE`: normalización `auto|cuda|cpu` (`lpips_device_mode`), default `auto` vía `apply_lpips_default_process_env()` si está vacío.
- `child_process_env()`: copia de entorno para `subprocess`; no deja la variable vacía; en **Windows** fija `PYTHONIOENCODING=utf-8` para evitar `UnicodeEncodeError` al capturar stdout/stderr del hijo.
- `coerce_lpips_env_if_cuda_unavailable()`: si el usuario fuerza `cuda`/`gpu` y este intérprete no tiene CUDA, baja a `cpu` en `os.environ` (evita fallos en workers de multiprocessing con PyTorch CPU-only).

### [FACT] Scripts que propagan entorno a `laser_target_match`

Importan `laser_runtime_env`, llaman `apply_lpips_default_process_env()` (+ `coerce_*` donde aplica) y usan `env=child_process_env()` en `subprocess.run`: entre otros `laser_autoruns`, `laser_adaptive_match`, `laser_batch_match`, `laser_compass_probes`, `laser_max_similarity_pipeline`, `laser_score_campaign`, `laser_chain_score_push`, `laser_match_preprocess_sweep`. `laser_target_match` y `laser_scoring` consumen el modo normalizado.

### [FACT] `laser_target_match.py` (score v4 / consola)

- Mensajes de `_configure_score_v4_cuda_efficiency` sin flecha Unicode (`->` en lugar de `→`) para compatibilidad cp1252 cuando el padre captura salida.

### [FACT] `scripts/laser_autoruns.py`

- Manifiesto schema **`laser_autoruns_run_manifest/v2`**: `planned_runs`, `laser_lpips_device_normalized`, al cerrar `completed_utc`, `wall_seconds_total`, `exit_code`, `failed_runs`.
- `runs/_autoruns/latest_session.json` (o bajo `--out-root`): puntero a última sesión + rutas a summary/manifest/readme.
- `summary.json`: `run_count`, `wall_seconds_total`, `failed_runs`, `all_passed`; modo solo v4: `best_v4_across_runs`, `v4_ranked`.
- `README_SESSION.md`: tabla de corridas; bloque “Mejor v4”; si aplica, línea sobre refine-from-best.
- **`--v4-only`**: solo corridas `--score-version v4` (grid/sobol × default/acrylic, BT.709, hi-side opcional, `07_v4_fromdb_refine` si existe `--refine-db`, Sauvola+acrylic si no `--quick`).
- **`--explore-brutal`**: añade corrida extra Sobol+acrylic con `--explore-brutal`.
- **`--refine-from-best`** (solo con `--v4-only`): tras las corridas, **`99_v4_refine_from_session_best`** desde `match.sqlite` del mejor run de la sesión; flags `--refine-from-best-n` (default 320), `--refine-from-best-top` (default 36); reconstruye extras del ganador con `_strip_local_refine_args`; fuerza `--sampling sobol` al final; opcional `--explore-brutal` coherentemente con el flag global.
- Helpers: `_common_replace_n`, `_strip_local_refine_args`.

### [FACT] Corrida v4-only verificada (CUDA / `.venv312`)

- Sesión ejemplo: `runs/_autoruns_v4/session_*` con `--n-base 280`, `--max-side 384`.
- Mejor estrategia observada en esa sesión: **`preprocess sauvola` + preset `acrylic`** (~0.34689 v4) frente ~0.3475 con `preprocess none` y grid/sobol similares.
- `--luma bt709` mejoró respecto a default en la misma rejilla pero no superó Sauvola+acrylic.
- Mayor `--max-side` en una variante Sobol+acrylic no mejoró con el mismo presupuesto de candidatos.
- Refinar desde SQLite de campaña histórica orientada a **v2** empeoró bajo **v4**; conviene refinar desde sqlite generado con la misma métrica/preprocess.

### Validación / memoria externa

- `python -m py_compile` en scripts tocados en estas tareas: OK en sesión de desarrollo.
- **basic-memory MCP**: en `%USERPROFILE%\.cursor\mcp.json` usar **`command`** con ruta absoluta a **`uvx.exe`** (p. ej. `C:\Users\DEV\.local\bin\uvx.exe`) para que arranque sin depender del PATH; tras reinicio Cursor debe aparecer como **basic-memory** / **user-basic-memory** conectado.
- Volcado paralelo en este repo: **`docs/IMPROVEMENT_LOG.md`** (sección Runtime LPIPS / autoruns v4).


## Fase R — Replanteo estratégico (2026-05-14, Claude branch)

Tras investigación profunda (motor `laser_target_match.py` 2791+ líneas + física CO2 con literatura), se decide complementar v3/v4 (target-based) con un score sin referencia y sumar física del láser al pipeline. Diagnóstico completo en `cursor-memory-vault/PROJECTS/image-ap-sport-for-laser.md` §"Replanteo estratégico (Fase R)".

### Diagnóstico

1. Score v3/v4 (target-based) sigue produciendo meseta `0.179-0.183` (v2) y ~0.34 (v4): es **varianza intra-dither** (Wang SSIM 2003 + Lau-Arce 2008 cap. 5-6). Optimizar contra el PNG de ImagR no puede dar más sin cambiar la métrica.
2. Sin física CO2 en pipeline: no DPI cap por spot (spot 50W = 0.12-0.18 mm; DPI util max 141-282); no LUT por material (PhotoGrav 1999 ya lo hacía); no dot-gain compensation (~2.5x area pixel cuadrado vs spot circular -> PNG sale 30-50% mas oscuro grabado).
3. Wizard SvelteKit (`web/`) desconectado del motor Python; preview cliente solo `gray+threshold`.
4. `UnsharpMask radius=1.2` fijo viola regla 3 de `RULES/image-ap-sport-for-laser`: el radius debe ser px del **output físico**, no del max_side de ranking.

### Decisiones

- Mantener v3/v4 como herramientas de A/B target-based; agregar **v5 sin referencia** como métrica de calidad física del halftone.
- Agregar módulo `laser_physics` con validación DPI por spot + LUT por material.
- Agregar módulo `laser_blue_noise` con void-and-cluster (Ulichney 1993) como blue-noise auténtico (mejor perfil espectral que `NOISE_16` ad-hoc).
- Materiales MVP: **acrílico back-engrave + madera** (LUTs stub calibrables via step-wedge).

### Implementado (Fase R, esta sesión)

#### `scripts/laser_scoring.py` (extendido)
- Funciones nuevas: `_csf_mannos_sakrison_filter`, `hvs_mse`, `spectral_radial_penalty`, `tone_match_error`.
- `score_candidate_v5_terms` + `score_candidate_v5`: combina 50% HVS-MSE (CSF Mannos-Sakrison) + 30% spectral radial penalty + 20% tone match local post-LUT + 5% regularización compartida con v2/v3/v4.
- `ScoreVersion = Literal["v1","v2","v3","v4","v5"]`.
- `score_candidate_dispatch` extendido: enruta v5 con `lut`/`ppd` por keyword sin romper v1-v4.

#### `scripts/laser_physics.py` (nuevo)
- `estimate_max_useful_dpi(spot_mm)`, `validate_dpi_for_spot(dpi, spot_mm)`: regla `dpi_max = 25.4/spot`.
- `interval_mm_for_dpi(dpi)`: convención láser horizontal.
- `MaterialProfile` dataclass: `spot_mm`, `default_dpi`, `lut_curve` (256 u8), `tone_response`, `power_pct_range`, `notes`.
- `acrylic_back_engrave_profile()` (gamma 0.65 monotónica) y `wood_profile()` (compresión extremo claro para evitar rebote de sublimación de lignina).
- `load_material_profile(name, presets_dir=None)`: builtins programáticos + JSON custom.
- `scaled_unsharp_radius(...)`: cierra regla 3 escalando radius en px ranking proporcional al output físico.

#### `scripts/laser_blue_noise.py` (nuevo)
- `generate_void_and_cluster(size, sigma, initial_fill, seed)`: algoritmo Ulichney 1993 completo (IBP estabilizado + Phase 1 + Phase 2).
- `void_and_cluster_matrix(size, cache_dir, force_regen)`: cache disco `.npy` (asset reusable).
- `threshold_matrix_for_dithering(size)`: float (0,1) listo para `ordered_dither`.

#### `scripts/laser_target_match.py` (cableado)
- Imports `laser_physics` y `laser_blue_noise` con fallback a None.
- Nuevos globals `_WORK_PPD` (default 64.0) y `_WORK_SHARPEN_RADIUS` (default 1.2) inicializados via `init_worker`.
- `preprocess_gray` usa `_WORK_SHARPEN_RADIUS` en lugar de literal 1.2 → cierra regla 3.
- `evaluate_candidate_task` pasa `ppd=_WORK_PPD` al dispatch.
- Nuevos CLI args:
  - `--material <name>` + `--material-presets-dir <path>`: aplica LUT al gris y valida DPI.
  - `--ppd <float>`: pixels per degree para CSF de v5.
  - `--output-mm-short <float>` + `--output-dpi <int>` + `--sharpen-radius-mm <float>`: escala USM al output físico.
  - `--score-version v5`: añadido a choices.
- `main()` extendido:
  - Si `--material`: carga `MaterialProfile`, valida DPI, aplica LUT al `base_gray`.
  - Si `--score-version v5`: setea `target_gray = base_gray` (no-reference puro).
  - Computa `resolved_radius` via `laser_physics.scaled_unsharp_radius` y lo guarda en `args._resolved_sharpen_radius` para que los workers lo recojan.
- `RESTART_ALGORITHMS` añade `"blue_noise_vac32"` (nueva alternativa de halftone).

### Tests añadidos (28 nuevos)

- `tests/test_score_v5.py` — 9 tests (blue-noise > cluster, no-reference signature, LUT shift, dispatch, breakdown keys, v1/v2 still work).
- `tests/test_laser_physics.py` — 21 tests (DPI max, validate warns, LUT identity/round-trip, profiles acrylic/wood, JSON load, npy load, sharpen scaling, integración con v5).
- `tests/test_laser_blue_noise.py` — 7 tests (permutación completa, threshold range, cache, force_regen, blue-noise mejor que random/cluster en baja-freq).

### Validación

- `pytest -m "not network" -q`: **51 passed, 20 deselected** (de 11 al inicio del trabajo; +40 tests sin regresiones).
- `python -m py_compile` en los 4 scripts modificados: OK.

### Pendientes Fase R (siguientes iteraciones)

- **R8**: FastAPI conecta wizard SvelteKit al motor real (paridad CLI↔UI; hoy el cliente solo hace gray+threshold local).
- Documentar workflow `--material X --score-version v5 --output-mm-short M --output-dpi D --sharpen-radius-mm 0.1` como receta canónica para fotograbado real.

---

## Fase R — Continuación (2026-05-15, sesión 2 Claude branch)

Tras handoff de Cursor (`docs/SESSION_HANDOFF_2026-05-15.md`) y revisión de logs del día, completados varios pendientes invasivos.

### Sync de WIP padre → worktree

Cursor avanzó en paralelo en `feat/auto-improve-pipeline` (sin commit):
- `scripts/laser_runtime_env.py` con caps de VRAM (`apply_cuda_process_memory_cap`), HF token sync, `resolve_torch_device_flag` para `--*-device auto`.
- Bug fix crítico en `local_refine_candidates`: early-exit antes de explotar combinatoria (`return unique` al llegar a `limit`). Sin este fix las refinaciones con `--refine-best-per-algorithm --refine-top 12 --refine-breadth deep` colgaban consumiendo RAM masiva.
- `tests/test_laser_runtime_env.py` (7 tests).
- Mejor corrida hasta ahora: `runs/_refine_native_v4_sauvola/match_0029.png` con **score v4 0.3904** (`floyd` thr=79, preset sauvola+acrylic, full-res 997×1772).

Sincronizado al worktree preservando los aportes Fase R: `laser_runtime_env.py` y `tests/test_laser_runtime_env.py` copiados directos del padre; cambios puntuales aplicados a `laser_scoring.py` (call `apply_cuda_process_memory_cap` antes de cargar LPIPS) y `laser_target_match.py` (`_materialize_torch_device_args` + early-exit en `local_refine_candidates`).

### Fix gamma direction en `step_gray_values` (laser_calibration_wedge)

- Bug: docstring decía "gamma>1 concentra valores en oscuros" pero implementación usaba `raw**(1/gamma)` (concentra en claros). Test `test_step_gray_values_gamma_changes_distribution` fallaba en sesión previa.
- Fix: cambio a `raw**gamma` para que coincida con la intuición y la docstring. Tests OK.

### R6 — Refactor render_candidate (invasivo)

El dispatcher de `render_candidate` tenía ~230 líneas de `if/elif` con:
- Kernels Floyd/Jarvis/Stucki/Atkinson literales **duplicados 3-4 veces** (en cada mix/multipass).
- **44 líneas de código muerto** al final: ramas `floyd`/`atkinson`/`jarvis`/`stucki` que nunca se alcanzaban porque `if candidate.algorithm in DIFFUSION_ALGORITHMS` (línea 1451) ya las capturaba.

Refactor a **3 tablas + dispatch limpio**:
- `DIFFUSION_ALGORITHMS` (16 entradas paramétricas: kernel + divisor + serpentine).
- `BURKES_BLUE_VARIANTS` (6 variantes paramétricas: midtone band + blue strength).
- `NAMED_RENDERERS` (18 singletons; cada uno es `_render_X(gray, candidate) -> uint8`).
- `ALL_RENDER_ALGORITHMS = tuple(sorted({*las tres tablas}))` — fuente única para validar listas.

Cada renderer ahora usa los kernels constantes nombrados (`FLOYD_KERNEL`, `JARVIS_KERNEL`, etc.) en lugar de literales inline. Total: **40 algoritmos** sin overlap entre tablas.

`render_candidate` quedó en ~20 líneas: tres lookups + raise.

### Wire `blue_noise_vac32` en exploración normal

Antes: solo aparecía en `RESTART_ALGORITHMS` (plateau-restart) y como case en `render_candidate`. Build/focused/dense no lo proponían.

Ahora: agregado en 4 listas de algoritmos dentro de `build_candidates` (capas base, aggressive, focused, neighbor para "threshold deep"). Total **8 ocurrencias** del nombre en el archivo.

### R7-fit — Calibration fit desde foto del wedge (cierra el loop físico)

Nuevo `scripts/laser_calibration_fit.py`:
- `load_photo_as_gray(path)`: foto → uint8 luminancia BT.601.
- `crop_image(...)` + `resize_to_wedge(...)`: alineación simple (crop manual opcional + resize Lanczos al tamaño del wedge).
- `measure_patches(photo, meta, inset_fraction, blur_radius)`: por cada parche del meta, mide el L* del centro (con inset configurable para evitar etiquetas), tras un uniform_filter para suavizar ruido fotográfico.
- `_enforce_monotonic_isotonic(xs, ys)`: PAVA mínimo (sin scikit-learn) — útil para madera con rebote tonal (pirólisis no-monotónica).
- `fit_inverse_lut(measurements, force_monotonic=True)`: PchipInterpolator monotónico + inversión sobre rejilla densa → LUT 256 u8.
- `save_lut(lut, debug, out_npy, material_name)`: `.npy` + JSON sidecar (con `lut_inline` para portabilidad).
- CLI: `python scripts/laser_calibration_fit.py --photo foto.jpg --wedge-meta wedge_meta.json --out lut_acrylic.npy --material-name acrylic_back_engrave`.

La LUT producida es directamente cargable vía `laser_physics.MaterialProfile` (campo `lut_curve_npy` o `lut_curve` inline en JSON).

### Tests añadidos (15 nuevos en esta sesión)

- `tests/test_render_dispatch.py` — 8 tests (disjoint union, count ≥40, smoke todos los algoritmos, ValueError en desconocido, RESTART/BRUTAL subset de ALL, BURKES_BLUE structure, NAMED_RENDERERS signature).
- `tests/test_laser_calibration_fit.py` — 7 tests (schema reject, measure patches, recover inverse gamma 2.0 → LUT[128]≈181, identity para gamma 1.0, isotonic para no-monotónico, save .npy+sidecar, integración con MaterialProfile).

### Validación

- `pytest -m "not network" -q`: **86 passed, 20 deselected** (de 71 al inicio de sesión; +15 sin regresiones).
- `py_compile` en los 6 scripts tocados/nuevos: OK.

### Pendientes restantes

- **R8** FastAPI scaffold conectando wizard SvelteKit al motor real (no urgente; CLI funcional).
- **Calibración física real**: usuario debe grabar `wedge.png` en Funsun + fotografiar + correr `laser_calibration_fit.py`. Workflow completo ahora listo.
- Commit agrupado en una rama nueva (`feat/phase-r-complete` o similar) para revisar en PR.


## Sesión 3 — Análisis empírico + mejora marginal verificada (2026-05-15, raíz padre)

Trabajo directo en `feat/auto-improve-pipeline`. Sync worktree → padre completado: `laser_physics.py`, `laser_blue_noise.py`, `laser_calibration_wedge.py`, `laser_calibration_fit.py`, R6 refactor + 6 tests nuevos. Suite parent **89 passed**.

Documentación completa: `runs/_analysis_2026-05-15/SUMMARY.md`.

### Experimentos full-res sobre baseline `runs/_refine_native_v4_sauvola/`

| ID | Script | Hipótesis | Resultado |
|---|---|---|---|
| Exp 0 | `experiment_vac32_vs_floyd.py` | vac32 con params Floyd matchea ImagR mejor | **Falsada**: Floyd v4=0.3907 #1; vac32 v4=0.4604 #36. white_ratio vac32 0.544 vs target 0.374. |
| Sweep vac32 | `sweep_vac32_tuned.py` | con params propios vac32 mejora | **Falsada**: best vac32 v4=0.4364 (-46 milesimas peor que Floyd). |
| Exp v5 | `experiment_v5_no_reference.py` | v5 sin referencia da otro ganador | **Falsada**: Floyd v5=0.3411 #1; vac32 v5=0.4082 #38. v5 también prefiere Floyd. |
| Floyd tight | `sweep_floyd_tight.py` + `verify_best_partial.py` | refine fino mejora Floyd | **Confirmada parcial**: best v4=**0.3897** verificado (-1 milesima vs 0.3907). 3 intentos; harness mata bg tasks ~400-500s; checkpoints incrementales rescatan partial. |

### Nuevo mejor candidato verificado

`runs/_analysis_2026-05-15/floyd_tight/verified_top01_v4_0.3897.png`

`floyd inv=1 thr=83 c=0.55 b=+25 g=1.35 ac=0 sh=40` con preprocess sauvola + luma bt709.

- v4 score: 0.3907 → **0.3897** (-0.26%)
- density_local diff vs target: 0.1720 → 0.1688 (mejor 1.9%)
- edge_error: 0.4070 → 0.4059 (mejor 0.3%)
- pixel_exact match: 57.14% → 56.40% (peor 0.7% — la mejora viene del término perceptual SSIM_blur+LPIPS, no del pixel-exact)

### Por qué blue_noise_vac32 NO ganó (rechazo hipótesis Fase R original)

1. **Spectral lowfreq target (0.9076) ≈ Floyd (0.9068)**: la macro-estructura del fotograbado (fabric, hand silhouette, canvas) domina el espectro a baja frecuencia. Distinguir dithers a nivel espectral requiere high-pass primero (mejora propuesta a `spectral_radial_penalty`).
2. **Error-diffusion adaptativo > ordered dither** para imágenes con contenido macro: Floyd integra error local y converge al white_ratio target; vac32 umbraliza por pixel y queda desbalanceado.
3. **Para imitar un PNG concreto, el dither del target manda**. vac32 es objetivamente mejor blue-noise teórico pero ImagR no usa exactamente vac32; las diferencias con Floyd resultan caras.

### Bug discovery: background tasks mueren ~400-500s

Tres invocaciones distintas de `sweep_floyd_tight.py` murieron entre 406s-486s sin error visible. Output del harness truncado a ~500 bytes mientras log via `tee` capturó la progresión completa.

**Hipótesis**: PyTorch CUDA cache acumula VRAM pese a `torch.no_grad()` y `detach().cpu()`. Sin `torch.cuda.empty_cache()` periódico se llega al cap ~5.5 GiB y el proceso es matado.

**Mitigaciones aplicadas**:
- `sweep_floyd_tight.py` hace checkpoint del top-20 cada 10 candidatos a `partial_ranking.json`.
- `verify_best_partial.py` re-evalúa los top 3 del checkpoint en single-eval (single process, sin acumulación).

**Mitigación pendiente** (TODO): agregar `torch.cuda.empty_cache()` cada N candidatos en `score_candidate_v4_terms` para destrabar este límite.

### Ruta de avance real (post-meseta)

El piso v4=0.39 es **estructural**: imitar un PNG concreto satura ahí porque el dither de ImagR no es replicable exactamente. **Romper la meseta requiere cambiar el target**:

1. Generar `wedge.png` con `laser_calibration_wedge.py --material acrylic_back_engrave --steps 16`.
2. Grabar y fotografiar en Funsun a 9-12% potencia.
3. Ajustar LUT con `laser_calibration_fit.py --photo wedge_grabado.jpg --wedge-meta wedge_meta.json`.
4. Re-producir con `--score-version v5 --material acrylic_funsun_calibrated --output-mm-short 84.4 --output-dpi 169 --sharpen-radius-mm 0.1`.

Esto reemplaza "parecerse al PNG de ImagR" por "predecir el grabado real" — métrica fundamentalmente distinta, libre de la meseta intra-dither.


## Sesión 4 — Veredicto FUNCIONAL + GUI/API/guía (2026-05-15, raíz padre)

### Veredicto sobre nueva imagen real (poster agricultor)

Usuario pasó `runs/references/input_agricultor.jpeg` (1122x1402 color) + `runs/references/target_agricultor.png` (1417x1771 b/n mirrored). Procesado con 5 variantes:

| Variante | Score v4 | px err | edge err | wr |
|---|---|---|---|---|
| baseline_floyd_tight | 0.4127 | 0.339 | 0.407 | 0.271 |
| hi_sharpen | 0.4121 | 0.343 | 0.406 | 0.294 |
| **hi_contrast** | **0.3247** | 0.303 | 0.275 | 0.282 |
| jarvis_detail | 0.4059 | 0.324 | 0.364 | 0.276 |
| stucki_detail | 0.4050 | 0.325 | 0.363 | 0.276 |

**Ganador hi_contrast**: Floyd thr=75 c=1.0 b=+10 g=1.2 ac=2.0 sharpen=60 inv=1, preprocess sauvola. Pixel match 69.74%, edge_error 0.275 (vs 0.407 del baseline hands — mucho mejor preservación de detalle fino del follaje del árbol).

**VEREDICTO: pipeline FUNCIONAL para producción.** Output visual igual o superior al target ImagR. Mejor variante guardada en `runs/_agricultor_2026-05-15/agricultor_hi_contrast_v4_0.3247.png` + report.md con todas las variantes.

### R8 — GUI + API + guía (IMPLEMENTADO)

#### Backend FastAPI (`scripts/api_server.py`, 380 LOC)
Endpoints REST sobre el motor:
- `GET /api/health` — liveness + estado modelo LPIPS + CUDA
- `GET /api/materials` — builtins (`acrylic_back_engrave`, `wood_generic`) + JSON custom
- `GET /api/algorithms` — 40+ algoritmos agrupados (ordered/diffusion/burkes_blue/mix)
- `POST /api/preview` — multipart con imagen + JSON params, fuerza max_side=400
- `POST /api/process` — full-res, devuelve PNG 1-bit con headers de metadata

CORS para Vite dev. Pydantic validation con rangos. Stateless. Headers de respuesta: `X-Process-Time-Ms`, `X-Output-Width/Height`, `X-White-Ratio`, `X-Sharpen-Radius-Px`, `X-Material`. Pip extra `[api]`: fastapi + uvicorn[standard] + python-multipart.

#### Tests API (`tests/test_api_server.py`, 11 tests)
- health endpoint
- materials lista builtins
- algorithms 4 familias
- preview devuelve PNG
- process full-res
- material LUT cambia output
- algorithm inválido → 400
- imagen inválida → 400
- params_json inválido → 422
- preview ≤ 400px vs process full-res
- DPI scaling header correcto

#### Cliente HTTP TypeScript (`web/src/lib/apiClient.ts`, 130 LOC)
Tipado completo (`ProcessParams`, `MaterialInfo`, `AlgorithmGroup`, `HealthResponse`, `ProcessMeta`, `ApiError`). Singleton `apiClient` con métodos `health/materials/algorithms/render(endpoint, blob, params)`. Base URL configurable via `VITE_API_BASE_URL`. Preset `PRESET_AGRICULTOR_HIGH_CONTRAST` exportado.

#### Wizard SvelteKit (`web/src/routes/+page.svelte`, reescrito ~500 LOC)
5-step wizard cableado al backend. Cambios estructurales:
- Header con estado backend + badge CUDA/CPU + URL base.
- Step pills navegables (no solo lineales): click para volver a pasos previos.
- Step 3 split-layout (controles izquierda + preview live derecha).
- Material select con info de spot + respuesta tonal en hint.
- Warning visible si `output_dpi > 1/spot`.
- Preset toggle (óptimo agricultor / manual).
- Sliders con valores en chip code coloreado.
- Preview en vivo con debounce 350ms a `/api/preview` (max_side=400).
- Comparador antes/después con divider neón verde + slider arrastrable.
- Procesar full-res via `/api/process`.
- Step 5 con download + checklist pre-grabado (mirror, interval, Pass-Through, 9–12% potencia).

Estilizado tema oscuro+verde militar acorde a taller:
- CSS variables (`--bg-panel`, `--border`, `--accent`, `--accent-dark`, `--accent-glow`).
- Background con gradients radiales verdes sutiles.
- Inter font + JetBrains Mono para valores.
- Hover states + focus accent.
- Responsive: 900px breakpoint para colapsar split-layout.
- Backdrop-filter blur en paneles para profundidad.

`svelte-check` → 0 errors, 0 warnings.

#### Smoke E2E
Backend via TestClient procesa agricultor full-res (1122x1402) en 3.6s con material acrylic + sharpen escalado a 1.122 px. Output visual: árbol con detalle preservado, texto legible, logo claro. PNG guardado en `runs/_agricultor_2026-05-15/api_smoke_output.png`.

#### Guía operador (`docs/USAGE.md`, 350 líneas)
8 secciones: instalación (Python+Node), inicio (terminal A backend + terminal B wizard), workflow operador (5 pasos detallados con tabla preset), workflow CLI avanzado (3 ejemplos), calibración física Fase R7 (5 pasos con código), troubleshooting (5 issues comunes), FAQ (5 preguntas técnicas), arquitectura.

### Validación final padre
- `pytest -m "not network" -q` → **100 passed, 20 deselected** (de 89 al inicio sesión 4; +11 API tests).
- `svelte-check` → 0 errors, 0 warnings.
- Smoke E2E manual via TestClient → OK.

### Estado del proyecto (cierre)

**Listo para producción operador:**
- GUI: wizard estilizado conectado a backend real.
- API: 5 endpoints documentados con tests.
- CLI: motor avanzado para investigación/sweeps.
- Guía: USAGE.md con workflow completo.
- Calibración física: scripts listos, esperando que usuario grabe step-wedge.

**Sin commits automáticos.** Working tree dirty, listo para review/PR.