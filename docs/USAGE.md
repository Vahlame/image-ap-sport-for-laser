# Image AP — Laser Image Prep · Guía de uso

Herramienta para preparar fotos para grabado láser CO2. Convierte una foto a color a un **PNG 1-bit listo para Pass-Through en LightBurn** (u otro CAM), con física del láser cableada (DPI cap por spot, LUT por material, sharpen escalado al output físico).

> **Cuándo usar la GUI:** si sos el operador del taller. Subís foto → ajustes por material → preview → descargás PNG.
>
> **Cuándo usar el CLI:** investigación, sweeps masivos, calibración, integración en pipelines.

---

## 1 · Instalación

Requiere Python ≥3.11 y Node ≥18.

```powershell
# Clonar repo (o ya estás dentro)
cd "C:\Users\DEV\Documents\GitHub\image ap sport for laser"

# Backend Python + API
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -e ".[api,perceptual,dev]"

# Frontend SvelteKit
cd web
npm install
cd ..
```

Opcional para SAM2 (segmentación avanzada):

```powershell
pip install -e ".[sam2]"
```

Opcional para CUDA (acelera score v4 LPIPS):

```powershell
# Instalar CUDA toolkit oficial de Nvidia (≥12.4) + driver compatible.
# Verificar:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`.env` (no commitear):

```env
LASER_LPIPS_DEVICE=auto
LASER_CUDA_MEMORY_CAP_GIB=5.5
HF_TOKEN=hf_...   # solo si usás SAM2 con modelos privados
```

---

## 2 · Iniciar la app (operador)

Abrí **dos terminales**.

**Terminal A — backend FastAPI:**

```powershell
.\.venv312\Scripts\Activate.ps1
uvicorn scripts.api_server:app --host 127.0.0.1 --port 8000 --reload
```

Debería decir `Application startup complete.` Verificá:

```powershell
curl http://127.0.0.1:8000/api/health
```

**Terminal B — wizard SvelteKit:**

```powershell
cd web
npm run dev
```

Abrí `http://localhost:5173` en el navegador.

En el header del wizard verás un punto **verde** y la etiqueta `CUDA` o `CPU` confirmando la conexión.

---

## 3 · Workflow operador (wizard web)

### Paso 1 — Subir
Arrastrá la foto a la zona de drop. Formatos: PNG, JPG, WebP. La resolución no importa — el pipeline resizéa al tamaño físico mm × DPI.

### Paso 2 — Recortar
Ajustá el recorte (cropper.js). El recorte se envía al backend tal cual.

### Paso 3 — Ajustes

**Preset óptimo** (recomendado por defecto): Floyd-Steinberg con params calibrados experimentalmente sobre poster de prueba `agricultor`:

| Param | Valor |
|---|---|
| Algoritmo | `floyd` |
| Pre-procesado | `sauvola` |
| Invertir | sí |
| Umbral | 75 |
| Contraste | 1.0 |
| Brillo | +10 |
| Gamma | 1.2 |
| Autocontraste | 2.0 |
| Sharpen | 60% |

Si el preset no encaja con tu foto, pulsá **Manual** y tocá sliders. El preview se actualiza en ~300 ms.

**Material físico:**
- `acrylic_back_engrave`: acrílico colado, cara posterior, baja potencia 9–12% (Funsun 50W). Aplica LUT gamma 0.65 para compensar dot-gain.
- `wood_generic`: madera genérica, respuesta tonal no-monotónica (LUT comprime extremo claro para evitar la zona de aclarado por sublimación de lignina).
- *Custom*: ver §5 (calibración física).

**Lado corto + DPI:** si los completás, el sharpen radius (USM) se escala automáticamente al tamaño físico real. Sin estos valores, queda en 1.2 px legacy.

> Si el DPI excede `1/spot` del material aparece warning. Para spot 0.15 mm máximo recomendado = 169 DPI. Más allá sólo solapa el haz.

### Paso 4 — Resultado (full-res)
Click **Procesar full-res**. Tarda ~1–5 s según resolución. Mostrá comparador antes/después con slider.

### Paso 5 — Descargar
Descarga `laser_ready_<material>_<algoritmo>.png`. PNG en mode `L` con valores estrictamente 0 y 255.

**Checklist pre-grabado** (incluido en la UI):
- [ ] Mirror (MirrorX) en LightBurn si es back-engrave en acrílico.
- [ ] `interval = 25.4 / DPI` mm en la capa imagen.
- [ ] **Pass-Through activado** en LightBurn (que NO re-dither el PNG).
- [ ] Probar 9–12 % potencia en acrílico antes del trabajo definitivo.

---

## 4 · Workflow CLI (avanzado)

Para investigación, sweeps masivos o integración programática. El motor real es `scripts/laser_target_match.py` con CLI completo.

**Single render con params concretos** (sin búsqueda):

```powershell
python scripts\laser_target_match.py `
  --input runs\references\input_agricultor.jpeg `
  --target runs\references\target_agricultor.png `
  --out runs\manual_test `
  --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 0 --n 1 --workers 1 `
  --material acrylic_back_engrave `
  --output-mm-short 100 --output-dpi 169 --sharpen-radius-mm 0.10
```

**Sweep masivo** (varios miles de combinaciones):

```powershell
python scripts\laser_target_match.py `
  --input foto.jpg --target ref.png `
  --out runs\sweep_acrylic `
  --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 384 --n 1500 --workers 1 --sampling sobol --guided-explore
```

**Refine seguro desde sqlite previa**:

```powershell
python scripts\laser_target_match.py `
  --input foto.jpg --target ref.png `
  --out runs\refine_safe `
  --preprocess-mode sauvola --score-version v4 --luma bt709 `
  --max-side 0 --n 150 --workers 1 `
  --refine-db runs\sweep_acrylic\match.sqlite `
  --refine-top 3 --refine-breadth normal --no-plateau-detect
```

> **No usar** `--refine-best-per-algorithm --refine-top 12 --refine-breadth deep` — bug histórico OOM/RAM (ver `docs/SESSION_HANDOFF_2026-05-15.md` §6). Si lo necesitás, primero corré con `--refine-breadth normal` y top bajo.

---

## 5 · Calibración física por material (Fase R7)

Romper la meseta de score que reproduce ImagR requiere **calibrar contra el grabado real**, no contra el PNG de ImagR. Workflow completo:

1. **Generar tira step-wedge**:
   ```powershell
   python scripts\laser_calibration_wedge.py `
     --out runs\wedge_acrylic.png `
     --material acrylic_back_engrave `
     --steps 16 --square-mm 10 --dpi 169 `
     --dither blue_noise_vac32
   ```
   Esto crea `wedge_acrylic.png` (PNG 1-bit listo para grabar) y `wedge_acrylic_meta.json` (posiciones + valores para el fit).

2. **Grabar la tira** en tu Funsun a 9–12 % potencia en acrílico, configurando DPI = 169 y Pass-Through.

3. **Fotografiar** bajo luz difusa cruzada (resalta microcavidades). Usa cámara fija, fondo neutro, sin reflejos.

4. **Ajustar la LUT** desde la foto:
   ```powershell
   python scripts\laser_calibration_fit.py `
     --photo wedge_grabado.jpg `
     --wedge-meta runs\wedge_acrylic_meta.json `
     --out presets\materials\acrylic_funsun_calibrated.npy `
     --material-name acrylic_funsun_calibrated
   ```

5. **Usar la LUT calibrada** en producción:
   - Crear `presets/materials/acrylic_funsun_calibrated.json`:
     ```json
     {
       "name": "acrylic_funsun_calibrated",
       "spot_mm": 0.15,
       "default_dpi": 169,
       "lut_curve_npy": "acrylic_funsun_calibrated.npy",
       "tone_response": "monotonic",
       "power_pct_range": [9.0, 14.0],
       "notes": "Calibrada con step-wedge real, Funsun 50W"
     }
     ```
   - Aparecerá automáticamente en el wizard (re-cargar `/api/materials`).

---

## 6 · Troubleshooting

### `Backend offline` en el wizard
- Verificar que `uvicorn scripts.api_server:app` esté corriendo.
- Revisar puerto: default 8000. Si lo cambiaste, setear `VITE_API_BASE_URL=http://127.0.0.1:NUEVO_PUERTO` en `web/.env`.

### Procesado lento (>10 s) para imágenes grandes
- Usá `max_side` en preview (la API ya lo fuerza a 400 para `/api/preview`).
- Si querés full-res rápido, reducí dimensiones físicas (mm) o DPI.
- CUDA acelera score v4 (LPIPS), no afecta render a full-res. El render full-res está en CPU.

### Score v4 muere en sweeps largos (>500 s)
- **Ya fixed** en `scripts/laser_scoring.py`: `torch.cuda.empty_cache()` cada 25 calls. Si persiste, agregá `--max-side 384` para reducir presupuesto.

### DPI warning
- Tu material tiene un spot físico. DPI > `1/spot` desperdicia tiempo de grabado sin ganar detalle. Bajá DPI o cambiá lente para reducir spot.

### Material `wood` produce zonas claras inesperadas
- La madera tiene respuesta **no-monotónica**: bajo cierto umbral oscurece, cruzado aclara (sublimación de lignina). Reducí potencia o calibrá con step-wedge real (sección 5).

### Tests fallan tras pull
```powershell
pip install -e ".[api,perceptual,dev]"
python -m pytest -m "not network" -q
```
Debe dar 100+ passed.

---

## 7 · FAQ

**¿Por qué Floyd-Steinberg y no `blue_noise_vac32` (Ulichney)?**
Empíricamente Floyd gana contra el target ImagR (sesión 3, ranking v4 #1 vs vac32 #36). La macro-estructura de la foto domina el espectro a baja frecuencia y vac32 no compensa esa diferencia.

**¿Qué es la meseta de score 0.39?**
Es el piso estructural de imitar el PNG de ImagR. Distintos dithers válidos producen patrones perceptualmente equivalentes pero distintos pixel-a-pixel. Para bajarla hay que cambiar el target (calibración física, sección 5).

**¿Cuándo usar v4 vs v5?**
- **v4** (target-based): comparás contra un PNG de referencia (ImagR, output previo). Útil para A/B entre params.
- **v5** (no-reference): calidad física del halftone sin depender de un PNG objetivo. Útil cuando ya tenés LUT material calibrada y querés evaluar el output contra el gris ideal.

**¿Por qué el PNG queda con texto al derecho y el target estaba mirrored?**
El target del usuario (ImagR) estaba pre-mirrored para back-engrave. Mi pipeline NO emite mirror — eso lo hace el operador en LightBurn (`MirrorX`). Esto es intencional y respeta la regla "última verificación en el CAM" (regla 5).

**¿Puedo usar la API sin la GUI?**
Sí. `curl` o cualquier cliente HTTP que mande multipart con `image` + `params_json`. Endpoints documentados en §1.4 de la sección 1, ejemplos en `tests/test_api_server.py`.

**¿Cómo agrego un material nuevo sin LUT calibrada?**
Crear `presets/materials/mi_material.json` con `lut_curve: [0,1,2,...,255]` (identidad) o un gamma stub. Aparecerá en el wizard.

---

## 8 · Arquitectura técnica (corta)

```
web/                       SvelteKit 5 wizard (Vite dev :5173)
  src/routes/+page.svelte  5-step wizard
  src/lib/apiClient.ts     cliente HTTP tipado
  src/lib/components/      CropStage.svelte (cropper.js)

scripts/                   motor Python
  api_server.py            FastAPI (:8000), endpoints health/materials/algorithms/preview/process
  laser_target_match.py    motor principal (40+ algoritmos, 9 preprocess, scoring v1..v5)
  laser_scoring.py         metricas v1..v5 (v4 LPIPS, v5 no-reference HVS+spectral+tone)
  laser_physics.py         MaterialProfile, DPI cap por spot, scaled_unsharp_radius
  laser_blue_noise.py      void-and-cluster Ulichney 1993
  laser_calibration_wedge.py  generador de tira step-wedge
  laser_calibration_fit.py    fit LUT desde foto del wedge grabado
  laser_runtime_env.py     LPIPS device, VRAM caps, HF token sync

tests/                     pytest, 100+ tests
assets/blue_noise_*.npy    matrices VAC cacheadas
docs/                      PLAN_*.md, IMPROVEMENT_LOG.md, SESSION_HANDOFF_*.md, este archivo
runs/_*/                   experimentos (gitignored)
runs/references/           inputs/targets canónicos (gitignored salvo refs)
```

**Stack:**
- Python: numpy, scipy, scikit-image, Pillow, opencv-headless. Opcionales: torch+lpips (perceptual), numba (JIT), transformers (SAM2), fastapi+uvicorn (API).
- Frontend: SvelteKit 5, Vite, cropperjs, TypeScript strict.
- Testing: pytest + fastapi TestClient + svelte-check.

---

*Última actualización 2026-05-15. Para historial técnico ver [`IMPROVEMENT_LOG.md`](IMPROVEMENT_LOG.md). Para plan continuable ver [`PLAN_2026-05-15_session3.md`](PLAN_2026-05-15_session3.md) y [`PLAN_GUI_R8.md`](PLAN_GUI_R8.md).*
