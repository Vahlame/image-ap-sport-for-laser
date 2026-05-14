# image ap sport for laser

Herramienta en desarrollo: preparación de imágenes para grabado láser (PNG).

## Pruebas con imágenes de stock públicas

Requieren red para la primera descarga (se reutiliza caché en el directorio temporal de la sesión de `pytest`).

```bash
pip install -e ".[dev]"
pytest tests/test_stock_images.py -v -m network
```

Sin red: `pytest -m "not network"` (omitirá las de stock si están marcadas; hoy las de stock están marcadas `network`).

## Barrido de parámetros (prueba y error offline)

Misma lógica de preview que la web (gris BT.601 + contraste + brillo + umbral), sobre **una** imagen, generando muchos PNG + una base **SQLite** + **JSONL** para ordenar y filtrar.

El modo por defecto es `smart`: no se queda atado a un solo umbral. Genera una escalera de umbrales neutra (`threshold_ladder`), suma anclas guiadas por histograma/Otsu (`histogram_anchor_grid`) y completa con variaciones estratificadas (`stratified_jitter`). Así podés comparar visualmente y consultar la BD por origen.

```bash
pip install -e .
python scripts/laser_parameter_sweep.py --input tu_foto.png --out runs/exp1 --n 300 --seed 7 --max-side 1024
```

Salida: `runs/exp1/sweep.sqlite`, `sweep_manifest.jsonl`, `sweep_0001.png` … Además genera `index.html` con miniaturas filtrables por estrategia, `thumbs/` con previews livianos y `contact_sheet.png` como hoja de contacto. Al final imprime candidatos con `white_ratio` cercano a 0.5 (solo heurística) y una muestra ordenada de la escalera de umbrales.

Consultas útiles:

```bash
sqlite3 runs/exp1/sweep.sqlite "SELECT threshold, white_ratio, output_file FROM runs WHERE source='threshold_ladder' ORDER BY threshold;"
sqlite3 runs/exp1/sweep.sqlite "SELECT threshold, contrast, brightness, source, white_ratio, output_file FROM runs ORDER BY ABS(white_ratio - 0.5) LIMIT 12;"
```

Si querés el comportamiento anterior, usá `--mode stratified`.

Por defecto la carpeta de salida se limpia de artefactos generados (`sweep_*.png`, SQLite, manifest, reporte) antes de crear un experimento nuevo. Para acumular en la misma base usá `--resume`.

## Búsqueda contra imagen objetivo

Para aproximar una salida de referencia (por ejemplo una imagen procesada por ImagR), usá `laser_target_match.py`. Prueba varios algoritmos (`threshold`, `bayer4/8`, `floyd`, `atkinson`, `jarvis`, `stucki`), mezclas con Bayer y pipelines multi-pasada (por ejemplo medios tonos con Floyd + sombras con Bayer, Bayer→Floyd, threshold→Jarvis). Guarda todos los PNG en una carpeta y ordena los mejores en `index.html`.

### Historial y dashboard (meta)

Ver `docs/META_SYSTEM.md`. Vista Streamlit opcional:

```bash
pip install -e ".[dashboard]"
streamlit run scripts/dashboard.py
```

```bash
python scripts/laser_target_match.py --input foto.png --target referencia.png --out runs/target_match --n 2000 --max-side 240 --top-report 300 --workers 4
```

Notas:

- `--n` puede subir a miles; `--workers` paraleliza por procesos.
- `--max-side` controla la resolución de ranking. Usá 200–400 para explorar miles rápido; luego re-renderizá pocos candidatos a mayor resolución.
- `match.sqlite` permite consultar por score, algoritmo y parámetros; `index.html` muestra el top visible y `contact_sheet.png` resume los mejores.
- GPU: CuPy puede ayudar a futuro en kernels vectorizados, pero Floyd/Jarvis/Stucki tienen dependencia secuencial por píxel. Para esta fase, multiprocessing acelera todos los algoritmos de forma más directa.

Para comparar varias imágenes contra el mismo objetivo, usá el runner batch:

```bash
python scripts/laser_batch_match.py --target referencia.png --input img1.png --input img2.png --out runs/batch_match --n 800 --max-side 240 --workers 4
```

Salida: `runs/batch_match/index.html` con una tarjeta por imagen, subcarpetas por input, y `batch_summary.json` con el mejor candidato de cada corrida.

### Variantes de entrada antes del dither

Cuando el score se estanca, conviene buscar mejor crop, rotación y tono **antes** del dither final. `laser_input_variants.py` genera muchas variantes de entrada, las pre-rankea contra el target con densidad/bordes, guarda SQLite/JSONL/HTML y deja una carpeta lista para `laser_batch_match.py`.

```bash
python scripts/laser_input_variants.py --input foto.png --target referencia.png --out runs/input_variants --mode aggressive --limit 48 --score-max-side 260
python scripts/laser_batch_match.py --target referencia.png --input-dir runs/input_variants --out runs/batch_variants --n 800 --max-side 240 --workers 4
```

Salida: `variants.sqlite`, `variants_manifest.jsonl`, `index.html`, `thumbs/` y `variant_*.png`. Esto mantiene cada intento visible y consultable sin mezclarlo con los resultados de dither.

## Interfaz web (SvelteKit)

Aplicación en la carpeta `web/`:

```bash
cd web
npm install
npm run dev
```

Abre la URL que muestra Vite (por defecto `http://localhost:5173`). Wizard en español: subir → **recorte con Cropper.js** → ajustes (mm, DPI, referencia dither) → comparador antes/después (preview local con umbral) → descarga PNG de preview.

Build estático: `cd web && npm run build` → salida en `web/build/`.
