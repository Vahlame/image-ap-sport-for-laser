# Plan de sesión Claude — 2026-05-15 (sesión 3)

Documento auto-contenido: si la sesión se interrumpe, otra puede continuar leyendo solo este archivo + los logs/memoria referenciados.

**Rama destino:** `feat/auto-improve-pipeline` (raíz padre, NO worktree).
**Objetivo del usuario:** "analiza todo, prueba, investiga, mejora, debugea, da mejor resultado que antes. Analizá las imágenes y mejorá hacia el objetivo." Implementar en raíz padre.

---

## Estado al inicio de esta sesión

- **Padre (rama actual):** tiene WIP de Cursor (autoruns, runtime_env, refine fix, score v2/v3/v4). 44 tests OK (per handoff). Mejor: `runs/_refine_native_v4_sauvola/match_0029.png` score v4 **0.3904** — Floyd inv=1 thr=79 c=0.57 b=+30.5 g=1.28 ac=0.5 sharp=40.
- **Worktree `claude/compassionate-joliot-a5e124`:** tiene mis adiciones Fase R: v5 no-reference, `laser_physics`, `laser_blue_noise` (void-and-cluster Ulichney), `laser_calibration_wedge`, `laser_calibration_fit`, refactor R6 de `render_candidate` (3 tablas + dispatch), 86 tests OK.
- **Worktree `claude/elastic-raman-14a217`:** otra rama de Claude (no inspeccionada esta sesión; asumir abandonada).

## Diagnóstico (de lo que vemos en el best report)

Top 10 candidatos del best run TODOS comparten `c=0.57 b=+30.5 g=1.28 ac=0.5 sharp=40` (solo varía threshold). **Patrón clásico de meseta de score**: el optimizador encontró un mínimo local en la familia Floyd y los vecinos no mejoran.

| Métrica | Valor |
|---|---|
| Score v4 total | 0.390440 |
| Pixel error | 0.428624 — **alto** |
| Edge error | 0.406952 — **alto** |
| White ratio | 0.373437 vs target 0.373897 — **excelente match tonal global** |

Conclusión: el match GLOBAL (tono) está calibrado; falta match LOCAL (texturas/bordes). Sugiere que pre-procesado (segmentación) y/o algoritmo (blue-noise vs Floyd, registro espacial) podría romper la meseta.

## Plan de trabajo (ordenado, retomable)

### Paso 1 — Sync worktree → padre (parar la divergencia)
- Copiar archivos nuevos del worktree al padre:
  - `scripts/laser_physics.py`
  - `scripts/laser_blue_noise.py`
  - `scripts/laser_calibration_wedge.py`
  - `scripts/laser_calibration_fit.py`
  - `tests/test_score_v5.py`
  - `tests/test_laser_physics.py`
  - `tests/test_laser_blue_noise.py`
  - `tests/test_laser_calibration_wedge.py`
  - `tests/test_laser_calibration_fit.py`
  - `tests/test_render_dispatch.py`
  - `assets/blue_noise_*.npy`
- Re-aplicar mis cambios a `scripts/laser_scoring.py` (agregar v5 + helpers) y `scripts/laser_target_match.py` (R6 refactor + imports + globals + init_worker extension + preprocess_gray scaled radius + render_candidate dispatch + CLI args + main material LUT + RESTART_ALGORITHMS + blue_noise_vac32 en build_candidates).
- Verificar con `pytest -m "not network" -q`. Esperado: ≥86 passed.

### Paso 2 — Análisis visual de target vs best
- Cargar `runs/references/target_imagr_acrylic.png` (997×1772, target ImagR).
- Cargar `runs/_refine_native_v4_sauvola/match_0029.png` (mejor actual).
- Computar diff map y heatmap por región:
  - Diferencia local (block 32×32) en density: dónde sobra/falta blanco.
  - Diferencia en edge_map: dónde el dither pierde bordes (siluetas) o los exagera (texturas).
  - SSIM local map.
- Guardar `runs/_analysis_2026-05-15/` con `diff.png`, `density_diff.png`, `edge_diff.png`, `report.md`.
- **Hipótesis a confirmar**: la diferencia se concentra en bordes finos del sujeto (siluetas de dedos) → el dither Floyd los suaviza más que el de ImagR.

### Paso 3 — Mejoras dirigidas según análisis
Cada una se ejecuta como corrida pequeña y se compara con el baseline 0.3904. Orden por impacto esperado:

**3.a — Refine con blue_noise_vac32 + edge-aware mixes**
- Justificación: void-and-cluster tiene perfil espectral isotrópico (sin worms direccionales que Floyd genera). Si el target es ImagR con su Blue Noise, esto debe alinearlos.
- Comando:
  ```
  python scripts/laser_target_match.py `
    --input runs/references/foto_objetivo_sin_procesar.jpeg `
    --target runs/references/target_imagr_acrylic.png `
    --out runs/_phase3a_vac32_refine `
    --preprocess-mode sauvola --score-version v4 --luma bt709 `
    --max-side 0 --n 200 --workers 1 --no-plateau-detect `
    --refine-db runs/_refine_native_v4_sauvola/match.sqlite `
    --refine-top 3 --refine-breadth normal
  ```
  Más unas corridas variantes que fuercen `blue_noise_vac32` como algoritmo principal.

**3.b — Registro espacial antes del scoring (`--register`)**
- Justificación: si la foto está descentrada/rotada vs target, edge_error sube artificialmente. `laser_registration` ya existe.
- Comando: igual al 3.a pero con `--register affine` o `--register ecc`.

**3.c — Pre-procesado SAM2 (segmentación del sujeto, fondo blanco)**
- Justificación: el target ImagR claramente tiene fondo blanco limpio; el input tiene fondo. Sauvola NO segmenta, solo umbraliza local.
- Comando con `--preprocess-mode sam2 --sam2-prompts runs/references/sam2_prompt_center_158x280.json` (existe).

**3.d — Score v5 (no-reference) en paralelo**
- Justificación: confirmar que v5 no se "pega" en la meseta de target-matching. Si v5 ranking da otro ganador, vale evaluarlo visualmente.
- Comando con `--score-version v5 --material acrylic_back_engrave --output-mm-short 84.4 --output-dpi 169 --sharpen-radius-mm 0.1`.

**3.e — Combo ganador propuesto**
- Sauvola + blue_noise_vac32 + register affine + v4 (target-based) — para máxima similitud con ImagR.
- Sauvola + blue_noise_vac32 + v5 + material acrylic — para máxima calidad física.
- Comparar mejor de cada corrida visualmente y por score.

### Paso 4 — Documentar y decidir
- Best report comparativo: tabla con baseline 0.3904 y cada experimento.
- Imagen ganadora visualmente, sea cual sea su número v4.
- Actualizar `IMPROVEMENT_LOG.md` y `cursor-memory-vault/SESSION_LOG.md`.
- Listar cómo continuar (R8 FastAPI + calibración física real con Funsun).

## Constraints / no-fail rules

- **NO `git commit` ni `push`** sin pedirme primero (preferencia global usuario).
- **NO `--refine-best-per-algorithm --refine-top 12 --refine-breadth deep`** — bug OOM histórico documentado (handoff §6).
- Workers `1` con v4 + CUDA (6 GB VRAM); más workers tira OOM en LPIPS.
- Subprocess en Windows necesitan `PYTHONIOENCODING=utf-8` (ya cableado via `laser_runtime_env`).
- `--score-version v5` requiere `--material` o el target_gray sigue siendo ImagR (no es bug, es modo "semi-reference").

## Archivos clave a leer si retomás esto

1. **Este archivo.**
2. `docs/SESSION_HANDOFF_2026-05-15.md` — handoff de Cursor con contexto operacional.
3. `docs/IMPROVEMENT_LOG.md` — historial técnico completo.
4. `cursor-memory-vault/PROJECTS/image-ap-sport-for-laser.md` — diagnóstico Fase R + decisiones.
5. `cursor-memory-vault/MEMORY-laser-snips.md` — física CO2 transversal.
6. `runs/_refine_native_v4_sauvola/best_report.md` — baseline a batir.
7. `scripts/laser_target_match.py` `_materialize_torch_device_args`, `_configure_score_v4_cuda_efficiency`, `local_refine_candidates` early-exit.

## Checkpoints (estado real al ejecutar cada paso, completar al avanzar)

- [ ] **Paso 1** sync worktree → padre completado.
- [ ] Tests verdes en padre tras sync (≥86 esperado).
- [ ] **Paso 2** análisis visual ejecutado, reporte en `runs/_analysis_2026-05-15/`.
- [ ] **Paso 3.a** vac32 refine: score = ___.
- [ ] **Paso 3.b** register affine: score = ___.
- [ ] **Paso 3.c** SAM2 preprocess: score = ___.
- [ ] **Paso 3.d** v5 paralelo: best = ___, comparado visualmente.
- [ ] **Paso 3.e** combo ganador: score = ___.
- [ ] **Paso 4** documentado en IMPROVEMENT_LOG + SESSION_LOG.

---

*Generado por agente Claude al inicio de sesión 3, 2026-05-15. Si retomas: leer §"Checkpoints" para saber dónde quedó.*
