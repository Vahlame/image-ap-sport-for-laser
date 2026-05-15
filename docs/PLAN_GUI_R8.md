# Plan R8 — GUI estilizada + API + guía de uso

Auto-contenido. Si se interrumpe la sesión, otro agente puede retomar leyendo solo este archivo + los logs/memoria referenciados.

**Trigger:** se ejecuta SOLO si la imagen agricultor procesada (input usuario) supera o iguala visualmente al target (verificada en sección "Veredicto" abajo).

## Resumen del flujo objetivo

```
[Usuario] foto color
      │
      ▼
[Wizard SvelteKit web/]   <── ya existe estructura (web/src/routes/+page.svelte: 5 pasos)
      │
      ▼  (subir → recortar → ajustes material/mm/DPI → procesar)
      │
[FastAPI /api/process]   <── NUEVO: backend wrapping del motor Python
      │
      ▼
[scripts/laser_target_match.py o pipeline directo]
      │
      ▼
[PNG laser-ready descargable]
```

## Componentes a construir

### 1. Backend FastAPI (`scripts/api_server.py` o `backend/main.py`)

Endpoints mínimos:

| Verbo | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | Liveness probe (devuelve `{status: "ok", model_loaded: bool}`) |
| GET | `/api/materials` | Lista builtins de `laser_physics.MaterialProfile` + presets custom de `presets/materials/` |
| POST | `/api/process` | Recibe `multipart/form-data` con `image` (file) + JSON params (material, mm, dpi, algorithm, threshold, contrast, brightness, gamma, sharpen). Devuelve PNG laser-ready 1-bit como `image/png`. |
| POST | `/api/preview` | Igual a `/api/process` pero a baja resolución (max-side 400) para preview rápido (<2s). |
| GET | `/api/algorithms` | Devuelve `ALL_RENDER_ALGORITHMS` de `laser_target_match.py` + grupo (diffusion/ordered/mix). |

Stack:
- `fastapi` + `uvicorn[standard]` (extra `[api]` en `pyproject.toml`).
- `python-multipart` para uploads.
- CORS habilitado para `http://localhost:5173` (dev Vite).
- Lifespan event: pre-carga del modelo LPIPS si v4 se usa (lazy ya en `laser_scoring._get_lpips_eval`).

Diseño del request body para `/api/process`:

```json
{
  "params": {
    "material": "acrylic_back_engrave",   // o "wood_generic" o "custom:<id>"
    "output_mm_short": 84.4,
    "output_dpi": 169,
    "algorithm": "floyd",                  // de ALL_RENDER_ALGORITHMS
    "threshold": 83,
    "contrast": 0.55,
    "brightness": 25.0,
    "gamma": 1.35,
    "autocontrast": 0.0,
    "sharpen": 40.0,
    "sharpen_radius_mm": 0.10,
    "invert": true,
    "preprocess_mode": "sauvola",          // o "none", "grabcut", "sam2"
    "score_version": "v4"                  // v4 target, v5 no-ref
  }
}
```

Response: PNG binario con headers `Content-Disposition: attachment; filename="laser_ready.png"` + custom header `X-Process-Time-Ms` para debugging.

### 2. Wizard SvelteKit (`web/`)

Estado actual (web/src/routes/+page.svelte): 5 pasos (subir → recortar → ajustes → resultado → descarga). El preview es solo client-side gray+threshold. Hay que:

a) **Conectar al backend** via fetch a `/api/preview` y `/api/process` en lugar del preview local.
b) **Reemplazar laserPreview.ts** por un cliente HTTP que llama al backend.
c) **Cargar materiales dinámicamente** del endpoint `/api/materials`.
d) **Cargar algoritmos** del endpoint `/api/algorithms` (no hardcoded en frontend).
e) **Mostrar el progreso** con un spinner durante `/api/preview` y barra durante `/api/process` (fetch streaming si la API lo soporta).

Mejoras de UI/UX (estilizado):

- **Tema oscuro acorde a taller** (verde militar + negro), con acento verde lima en CTAs.
- **Tipografía**: Inter o sistema (sans-serif clean) en cuerpo; mono para valores numéricos.
- **Layout responsive**: laptop 1366 mín, mobile-friendly.
- **Accesibilidad**: foco visible, ARIA labels, contraste ≥ AA.
- **Step indicator** (1.../5) sticky en top.
- **Antes/Después** con slider horizontal arrastrable + zoom 2x con lupa.
- **Sliders de ajuste fino** con debounce 300ms para no spamear preview.

Stack ya está (Svelte 5, cropperjs, vite). Nada nuevo a instalar.

### 3. Guía de uso (`docs/USAGE.md`)

Cubrir:

1. **Instalación**: `pip install -e ".[api,dashboard,perceptual]"` + `cd web && npm install`.
2. **Iniciar backend**: `uvicorn scripts.api_server:app --reload --host 127.0.0.1 --port 8000`.
3. **Iniciar wizard**: `cd web && npm run dev` (Vite levanta en 5173).
4. **Workflow operador** paso a paso con screenshots.
5. **Workflow de calibración** (Fase R7): generar wedge → grabar → fotografiar → fit LUT → producción.
6. **Workflow CLI avanzado** (poder, sweeps, refines): solo párrafo apuntando a `IMPROVEMENT_LOG.md`.
7. **Troubleshooting**: GPU sin CUDA, OOM v4, qué hacer si DPI > spot, materiales custom.
8. **FAQ**: ¿por qué Floyd y no blue-noise? ¿qué es la meseta de score? ¿cuándo usar v4 vs v5?

### 4. Tests de la API

- `tests/test_api_server.py` con TestClient de FastAPI:
  - `test_health` → 200
  - `test_materials_lists_builtins` → contiene `acrylic_back_engrave` y `wood_generic`
  - `test_algorithms_lists_all` → ≥ 40 nombres
  - `test_process_minimal` → POST con imagen sintética + params válidos → PNG 1-bit
  - `test_process_validation_errors` → 422 si params fuera de rango
  - `test_preview_faster_than_process` → max-side 400 vs full
- Smoke E2E (manual o playwright si el budget lo permite).

## Constraints / no-fail

- **NO romper** los scripts CLI existentes — la API es ADITIVA, no reemplaza.
- **NO commit** automático.
- **CUDA está instalado** (usuario confirmó): default `--score-version v4` con GPU OK.
- API debe correr en **single process** (LPIPS lazy load + VRAM cap ya cableado en `laser_scoring._get_lpips_eval` con `empty_cache` cada 25 calls).
- El `/api/process` debe respetar la regla 3 (sharpen escalado al output) — ya está en `preprocess_gray` via `_WORK_SHARPEN_RADIUS`; el handler debe pasarlo bien.

## Orden de implementación

1. ✅ **Veredicto** sobre la imagen agricultor (HACE FALTA recibir las imágenes).
2. **Backend FastAPI** (~250 LOC, 1-2 hrs).
3. **Tests del backend** (~150 LOC, 30 min).
4. **Conectar Svelte al backend** (~100 LOC en `+page.svelte` + nuevo `apiClient.ts`).
5. **Estilizar** (tema oscuro + verde, layout responsive, slider antes/después decente) (~200 LOC CSS + 50 LOC componente lupa).
6. **`docs/USAGE.md`** con screenshots (~400 líneas).
7. **Smoke E2E**: server + wizard + procesar una imagen real, verificar PNG descargable.
8. Update `IMPROVEMENT_LOG.md` y `cursor-memory-vault/SESSION_LOG.md`.

## Veredicto pendiente (sección que rellena el agente cuando proceda)

> **NO COMPLETADO**: requiere imágenes nuevas (agricultor) en disco.

Cuando estén:
- Procesar input agricultor con baseline Floyd-tight params (`thr=83 c=0.55 b=+25 g=1.35 ac=0 sh=40 inv=1 floyd` con sauvola).
- También con v5 + material acrylic.
- Comparar contra el target del usuario.
- Decidir:
  - ✅ Si ≥ visualmente acceptable → **funcional**, proceder con FastAPI/GUI.
  - ❌ Si claramente inferior → identificar qué falla (algoritmo wrong, params wrong, registro), iterar antes de GUI.

---

*Generado al inicio de tarea GUI/API, 2026-05-15. Para continuar después del corte: leer §"Orden de implementación" y arrancar desde el primer item incompleto.*
