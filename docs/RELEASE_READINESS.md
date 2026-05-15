# Release readiness — 2026-05-15

Evaluacion para el siguiente agente / release humano. Integra trabajo **Cursor** (GPU, refine, tests) + trabajo **Claude** (API, fisica, GUI, v5, calibracion).

---

## Veredicto corto

| Tipo de release | Listo? |
|-----------------|--------|
| **Uso local en taller (doble clic + wizard)** | **Si**, con `Iniciar_Laser_App.bat` tras `Instalar_Dependencias.bat` |
| **Release git versionado (tag v0.2)** | **Casi** — falta commit limpio, changelog, sin secretos en repo |
| **Instalador .exe unico sin Python/Node** | **No** — requiere PyInstaller/Electron o similar (no implementado) |
| **Publico / tienda** | **No** — sin auth en API, CORS abierto, sin firma de codigo |

---

## Que aporto Claude (ya en el working tree)

| Area | Archivos / notas |
|------|------------------|
| **API FastAPI** | `scripts/api_server.py` — health, materials, algorithms, preview, process |
| **Fisica laser** | `scripts/laser_physics.py`, `laser_simulator.py` |
| **Blue noise / calibracion** | `laser_blue_noise.py`, `laser_calibration_wedge.py`, `laser_calibration_fit.py` |
| **Score v5 (tests)** | `tests/test_score_v5.py` + cambios en `laser_scoring.py` |
| **Render dispatch** | `tests/test_render_dispatch.py` |
| **GUI** | `web/src/lib/apiClient.ts`, `+page.svelte` conectado al backend |
| **Docs** | `docs/USAGE.md`, `docs/PLAN_GUI_R8.md`, `docs/PLAN_2026-05-15_session3.md` |
| **Tests API** | `tests/test_api_server.py` — 20 tests |

Worktrees en `.claude/worktrees/` son copias de trabajo; **la fuente de verdad es la raiz del repo** (rama `feat/auto-improve-pipeline`).

---

## Validacion ejecutada (esta sesion)

```
pytest tests -m "not network"   -> 101 passed, 11 skipped
pytest tests/test_api_server.py -> 20 passed (incluido en suite)
npm run build (web/)            -> OK (adapter-static -> web/build/)
```

Extras instalados en venv: `[api, perceptual, dev, sam2]`.

---

## Que aporto Cursor (misma rama)

- `laser_runtime_env.py` — GPU cap 5.5 GiB, HF token, workers v4
- Fix `local_refine_candidates` early-exit (evita OOM)
- `docs/SESSION_HANDOFF_2026-05-15.md`
- Corridas experimentales en `runs/_run_native_v4_sauvola`, `runs/_refine_native_v4_sauvola`

---

## Bloqueadores antes de "release oficial"

1. **Git:** muchos archivos sin commit; rama no mergeada a `main`.
2. **Secretos:** `.env` con HF_TOKEN debe quedar solo local (ya en `.gitignore`).
3. **No commitear:** `WhatsApp Image*.jpeg`, `assets/` pesados, `reports/` si son enormes — revisar `.gitignore`.
4. **Perceptual opcional:** torch+LPIPS ~5 GB en venv; primera corrida v4 descarga pesos Alex.
5. **SAM2 opcional:** transformers + modelos HF; no obligatorio para wizard basico.
6. **Seguridad API:** solo localhost; no exponer `0.0.0.0` sin auth.
7. **CLI masivo** (`laser_target_match --n 10000`) sigue siendo herramienta de investigacion, no del operador.

---

## Como arrancar (operador)

1. `Instalar_Dependencias.bat` (una vez)
2. Copiar `.env.example` -> `.env` (opcional HF_TOKEN)
3. **`Iniciar_Laser_App.bat`** (doble clic)
4. Navegador: http://localhost:5173
5. Al terminar: `Detener_Laser_App.bat`

Manual detallado: `docs/USAGE.md`.

---

## Checklist pre-tag v0.2.0 (sugerido)

- [ ] `git add` selectivo (sin .env, sin runs pesados, sin .venv312)
- [ ] Commit mensaje: feat: API + GUI wizard + physics/calibration + runtime GPU
- [ ] Tag `v0.2.0`
- [ ] Probar `Iniciar_Laser_App.bat` en maquina limpia
- [ ] Anotar en README enlace a `docs/USAGE.md` y lanzadores `.bat`

---

## Mejor resultado experimental (CLI, no GUI)

Referencia para calidad de matching contra target acrylic:

| Run | Score v4 | Archivo |
|-----|----------|---------|
| Refino | **0.3904** | `runs/_refine_native_v4_sauvola/match_0029.png` |
| Nativo | 0.3949 | `runs/_run_native_v4_sauvola/match_0089.png` |

El wizard usa preset **agricultor** (otro par de imagenes en `runs/references/`).
