# Handoff de sesion — 2026-05-15

Documento para el siguiente agente (Claude u otro): que se hizo, que fallo, que quedo en disco y como seguir sin repetir errores.

**Rama:** `feat/auto-improve-pipeline` (cambios locales, muchos sin commit).  
**Entorno:** Windows, Python `.venv312`, PyTorch `2.6.0+cu124`, GPU **RTX 3060 Laptop ~6 GB VRAM**.  
**Referencias canonicas:** `runs/references/foto_objetivo_sin_procesar.jpeg`, `runs/references/target_imagr_acrylic.png` (target ~997x1772).

---

## 1. Resumen ejecutivo

| Tema | Estado |
|------|--------|
| Tests automaticos | **44 passed** (`pytest tests`) |
| Mejor corrida util full-res | `runs/_refine_native_v4_sauvola/` score v4 **0.3904** |
| Corrida base full-res | `runs/_run_native_v4_sauvola/` score v4 **0.3949** |
| Bug critico refine-db | **Corregido** (explosion combinatoria RAM) |
| Token HF | En `.env` local (gitignored); ver `.env.example` |
| Limpieza disco | **74 carpetas** `runs/` borradas (~1,2 GB) + pip cache ~3,1 GB |
| Incidente RAM/disco | Corrida refine atascada consumio RAM masiva + swap (no ~150 GB en repo) |

---

## 2. Cambios de codigo en esta sesion

### `scripts/laser_runtime_env.py` (nuevo/ampliado)

- `LASER_LPIPS_DEVICE` default `auto`; `child_process_env()` para subprocess.
- `resolve_torch_device_flag()` — `cpu` / `cuda` / `auto` para DeepLab/U-Net/SAM2.
- `infer_v4_max_gpu_workers_cap()` — tope heuristico de workers LPIPS en GPU segun VRAM (6 GB -> **1 worker**).
- `apply_cuda_process_memory_cap()` — techo **~5.5 GiB** por proceso PyTorch si GPU reporta <= ~7.5 GiB (`LASER_CUDA_MEMORY_CAP_GIB`).
- `sync_hf_hub_token_env()` — unifica `HF_TOKEN` y `HUGGING_FACE_HUB_TOKEN`.

### `scripts/laser_target_match.py`

- `_configure_score_v4_cuda_efficiency` — cap workers + desactiva recycle de hijos con v4+CUDA.
- `--deeplab-device`, `--unet-device`, `--sam2-device` aceptan **`auto`**.
- `_materialize_torch_device_args()` tras parse.
- Aviso si falta token HF antes de SAM2.
- **`local_refine_candidates`:** early-exit al llegar a `--n` (antes materializaba millones de `Candidate` -> OOM).

### `scripts/laser_scoring.py`

- `apply_cuda_process_memory_cap(quiet=True)` antes de cargar LPIPS en workers.

### `tests/test_score_regression.py`

- Tests de `score_candidate_dispatch` vs v2/v3/v4 directo.
- Tests de errores (version desconocida, candidate None).

### `tests/test_laser_runtime_env.py` (nuevo)

- Flags torch/HF/CUDA cap.

### `pyproject.toml`

- Extra opcional `[sam2]` = transformers + accelerate.

### Otros

- `.env.example` — plantilla env (sin secretos).
- `.gitignore` — `.venv312/`.
- `README.md` — GPU 6 GB, SAM2, HF token, techo VRAM.
- `runs/references/sam2_prompt_center_158x280.json` — caja para input ~158x280 con `--max-side 280`.

---

## 3. Validacion ejecutada

```
pytest tests -v          -> 44 passed
npm run check (web/)     -> 0 errors
py_compile scripts/*.py  -> OK
```

Smoke CLI: `laser_target_match` con PNGs sinteticos, n=4, OK.

---

## 4. Experimentos y resultados (score v4, menor = mejor)

Metrica **v4** no comparable numericamente con v2/v3 (incluye LPIPS).

### Resolucion baja (~158x280, `--max-side 280`)

| Carpeta | Preprocess | n | Mejor score | Notas |
|---------|------------|---|-------------|-------|
| `_trial_v4_sauvola_bt709` | sauvola | 80 | **0.3508** | ~11 s total ranking |
| `_trial_v4_deeplab` | deeplab auto cuda | 48 | 0.3721 | |
| `_trial_sam2_hf` | sam2 tiny + HF token | 24 | 0.3958 | SAM2 ~0.46 s GPU |

### Full-res nativo (`--max-side 0` -> **997x1772**)

| Carpeta | Config | n | Tiempo | Mejor |
|---------|--------|---|--------|-------|
| `_run_native_v4_sauvola` | sauvola, bt709, workers 1 | 120 | ~652 s | **0.3949** `match_0089` `floyd_midtones_bayer_shadows` thr=65 |

### Refino desde SQLite base

| Carpeta | Config | Resultado |
|---------|--------|-----------|
| Intento 1 (background, **FALLO**) | `--refine-top 12 --refine-best-per-algorithm --breadth deep --n 400` | Colgado ~71 min generando candidatos; proceso matado; exit 4294967295 |
| `_refine_native_v4_sauvola` (**OK**) | Refino posterior (200 evals) | **0.3904** `match_0029` `floyd` thr=79; white_ratio casi igual al target |

**Ganador actual para revisar visualmente:**  
`runs/_refine_native_v4_sauvola/match_0029.png` vs `runs/references/target_imagr_acrylic.png`  
Galeria: `runs/_refine_native_v4_sauvola/index.html`

---

## 5. Por que v4 es lento y la GPU parece "parada" (NORMAL)

Por candidato a **full-res** (~1.7M px), medicion aproximada:

| Paso | Tiempo | Donde |
|------|--------|-------|
| `render_candidate` (dither Floyd/multipass) | **~3.7 s** | **CPU** (secuencial) |
| `score_candidate_v4` (blur, SSIM, bordes, densidad) | ~0.4-4 s | **Mayormente CPU** |
| LPIPS Alex | fraccion pequena | **GPU** (racha corta) |

Con `--workers 1` (recomendado en 6 GB VRAM) no hay paralelismo de evaluacion.  
Cada candidato escribe `match_XXXX.png` -> I/O disco.

SAM2: **una** inferencia GPU al inicio; el bucle de miles de candidatos es el mismo pipeline v4.

---

## 6. Incidente: refine-db + RAM 32 GB + disco 100%

### Causa

`--refine-db` + `--refine-top 12` + **`--refine-best-per-algorithm`** + **`--refine-breadth deep`**:

- SQLite tenia **38 algoritmos** -> hasta **12 anclas por algoritmo** = cientos de anclas.
- Por ancla, producto cartesiano **~50M combinaciones** (umbrales x tonos x algoritmos vecinos).
- Codigo **antiguo** acumulaba todo en RAM antes de truncar a `--n`.

Efecto: RAM llena, **swap/pagefile** al maximo, disco al 100%, GPU idle (aun no se evaluaba nada).

### Fix aplicado

`local_refine_candidates` hace **return** en cuanto `len(unique) >= limit` (--n).

### Reglas para el proximo agente

**NO usar:**

```text
--refine-best-per-algorithm --refine-top 12 --refine-breadth deep
```

**SI usar refino seguro:**

```text
--refine-db runs/_run_native_v4_sauvola/match.sqlite
--refine-top 3 --refine-breadth normal --n 150 --max-side 0
--workers 1 --score-version v4 --preprocess-mode sauvola --luma bt709
```

---

## 7. Disco: que se limpio y que quedo

### Liberado en sesion

- **74 carpetas** bajo `runs/` (experimentos viejos, smokes, campanas): **~1,18 GB**
- **`pip cache purge`:** **~3,1 GB**
- `__pycache__`, `.pytest_cache` en repo

### Conservado en `runs/`

| Carpeta | Contenido |
|---------|-----------|
| `references/` | fotos target/input + JSON SAM2 |
| `_run_native_v4_sauvola/` | corrida nativa 120 candidatos + reports |
| `_refine_native_v4_sauvola/` | refino 200 candidatos + reports |

El repo completo ~6 GB (incluye `.venv312` ~5 GB). **No explica ~150 GB** en disco; el pico fue **paginacion Windows** durante OOM. Recomendacion: **reiniciar PC** si el espacio libre no cuadra tras limpieza.

### Disco C: (al cierre de sesion)

~**381 GB libres** de ~951 GB total.

---

## 8. Variables de entorno utiles

Ver `.env.example`. Minimo recomendado en `.env` (no commitear):

```env
LASER_LPIPS_DEVICE=auto
LASER_CUDA_MEMORY_CAP_GIB=5.5
HF_TOKEN=hf_...   # solo lectura; crear en huggingface.co/settings/tokens
```

Cargar en PowerShell antes de scripts:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
    Set-Item -Path "env:$($matches[1].Trim())" -Value $matches[2].Trim()
  }
}
```

---

## 9. Comandos listos para copiar

### Ranking rapido (buscar)

```powershell
python scripts/laser_target_match.py `
  --input runs/references/foto_objetivo_sin_procesar.jpeg `
  --target runs/references/target_imagr_acrylic.png `
  --out runs/_search_280 `
  --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 280 --n 80 --workers 1 --no-plateau-detect
```

### Full-res solo top-K desde SQLite

```powershell
python scripts/laser_target_match.py `
  --input runs/references/foto_objetivo_sin_procesar.jpeg `
  --target runs/references/target_imagr_acrylic.png `
  --out runs/_fullres_top5 `
  --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 0 --from-db runs/_refine_native_v4_sauvola/match.sqlite `
  --from-db-top 5 --workers 1
```

### Refino seguro

```powershell
python scripts/laser_target_match.py `
  --input runs/references/foto_objetivo_sin_procesar.jpeg `
  --target runs/references/target_imagr_acrylic.png `
  --out runs/_refine_safe `
  --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 0 --n 150 --workers 1 `
  --refine-db runs/_refine_native_v4_sauvola/match.sqlite `
  --refine-top 3 --refine-breadth normal --no-plateau-detect
```

---

## 10. Tareas background / interrumpidas

| Task | Que era | Resultado |
|------|---------|-----------|
| refine 400 + best-per-algo + deep | Primer refino automatico | **Fallo** — OOM/colgado en generacion candidatos |
| pip cache + uv prune | Limpieza disco | pip **OK** (~3 GB); uv prune **no termino** (cache en uso) |

---

## 11. Pendientes sugeridos (no hechos en esta sesion)

1. Revisar **visualmente** `match_0029.png` vs target acrylic.
2. Refino seguro extra si el score v4 aun no convence visualmente.
3. Ajustar `sam2_prompt_center_158x280.json` si se usa SAM2 a otra resolucion.
4. Commit agrupado en `feat/auto-improve-pipeline` (runtime env, refine fix, tests, docs).
5. **Revocar/rotar HF token** si hubo exposicion en chat (usuario indico cuenta personal).
6. `uv cache prune --force` cuando no haya procesos uv.

---

## 12. Claude + release (2026-05-15 tarde)

- Integracion Claude en working tree: **API** (`api_server.py`), **GUI** (`apiClient.ts`), **fisica/calibracion**, score **v5** (tests).
- **101+ tests** offline OK; **web build** OK.
- Lanzadores Windows: `Instalar_Dependencias.bat`, **`Iniciar_Laser_App.bat`**, `Detener_Laser_App.bat`.
- Veredicto release: **`docs/RELEASE_READINESS.md`** — listo para taller local, no para instalador .exe unico.

---

## 13. Archivos clave para leer primero

1. Este documento.
2. `docs/IMPROVEMENT_LOG.md` (historial tecnico previo).
3. `runs/_refine_native_v4_sauvola/best_report.md`
4. `scripts/laser_runtime_env.py`
5. `scripts/laser_target_match.py` — `local_refine_candidates`, `_configure_score_v4_cuda_efficiency`

---

*Generado al cierre de sesion Cursor 2026-05-15. No incluye secretos; token HF solo en `.env` local.*
