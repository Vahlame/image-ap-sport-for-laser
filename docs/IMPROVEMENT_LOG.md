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

- Carpeta sin `.git` en este entorno: inicializar repo, rama `feat/auto-improve-pipeline`, commits incrementales y `git push` cuando exista `origin`.
- Corridas largas de smoke (`--n` 100 v2 + register) no re-ejecutadas aqui por tiempo; validar localmente con referencias reales.
- `PlateauDetector` sin accion `ABORT` (solo `RESTART` / `NONE`); ampliar si hace falta criterio de aborto global.

### Ajuste Sobol + git meta

- `dense_blue_family_candidates`: muestra Sobol con `n` potencia de 2 y recorte para evitar `UserWarning` de scipy.
- `.gitignore`: `runs/_meta/*.sqlite` para no versionar bases locales generadas por corridas.
