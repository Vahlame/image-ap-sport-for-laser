# Contribuir a Image AP — Laser Image Prep

Bienvenidos PRs. La licencia es **GPL-3.0-or-later**: podés modificarlo, distribuirlo,
usarlo comercialmente, pero los trabajos derivados deben **permanecer abiertos bajo la
misma licencia** y **preservar los créditos** de los autores originales.

## En 30 segundos

1. Fork → branch → PR.
2. Añadí tu línea en [`AUTHORS.md`](AUTHORS.md).
3. Tests verde (`pytest -m "not network" -q` + `cd web && npm run check`).
4. Una idea por PR. PRs chicos pasan más rápido.

## Requisitos

| | |
|---|---|
| Python | ≥ 3.11 |
| Node | ≥ 18 |
| OS | Windows / Linux / macOS (CI no garantizado en macOS) |
| GPU | Opcional (acelera score v4 LPIPS) |

## Setup local

```powershell
git clone https://github.com/<tu-fork>/image-ap-sport-for-laser
cd image-ap-sport-for-laser
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -e ".[api,perceptual,dev]"
cd web; npm install; cd ..
```

## Ejecutar tests

```powershell
# Suite Python (back + presets + simulador + API)
python -m pytest -m "not network" -q
# Esperado: 130+ passed

# Frontend type-check
cd web; npm run check; cd ..
```

## Estilo

- **Python**: 4 espacios, docstrings cortas, tipos donde aporte. Sin `from x import *`.
  Variables descriptivas en español o inglés según contexto del módulo.
- **TypeScript**: strict. Sin `any` salvo en boundaries con FFI/forms.
- **Mensajes de commit**: imperativo + objeto + por qué. Conventional Commits opcional.
- **Sin emojis** en código ni docs salvo que el archivo ya los use por convención.

## Qué se acepta

- Algoritmos de halftone nuevos (con tests de regresión).
- Materiales nuevos (`MaterialProfile` en `laser_physics.py` + LUT calibrada).
- Mejoras al auto-detector de presets (con casos de prueba).
- Localización de la UI (i18n).
- Bug fixes con test que demuestra el bug + fix.
- Documentación, ejemplos, tutoriales.
- Performance: micro-benchmarks antes/después.

## Qué NO se acepta

- Código sin tests.
- Cambios masivos sin discusión previa (abrí un Issue primero).
- Dependencias propietarias o con licencia incompatible con GPL-3.
- Algoritmos copiados literal de software propietario (PhotoGrav, ImagR, etc.).
  Las matemáticas públicas + implementación propia son OK; copiar binarios o pesos no.

## Cómo agregar un algoritmo de halftone

1. Implementá la función en `scripts/laser_target_match.py` siguiendo el patrón:
   ```python
   def _render_mi_algoritmo(gray: np.ndarray, candidate: Candidate) -> np.ndarray:
       # ... lógica ...
       return binary  # uint8 {0, 255}
   ```
2. Registrá en `NAMED_RENDERERS` o agregá kernel a `DIFFUSION_ALGORITHMS`.
3. Agregá test en `tests/test_render_dispatch.py` que verifique output binario válido.
4. Si querés que aparezca en exploración masiva, agregá nombre a `RESTART_ALGORITHMS`.
5. Actualizá `CHANGELOG.md` bajo `[Unreleased]`.

## Cómo agregar un material

1. En `scripts/laser_physics.py`, agregá función `mi_material_profile() -> MaterialProfile`.
2. Registrá en `_builtin_profiles()`.
3. Agregá test en `tests/test_laser_physics.py` que verifique campos sanos.
4. Si tenés LUT calibrada físicamente, ponela bajo `presets/materials/<nombre>.json`.

## Cómo agregar un preset

1. En `scripts/laser_presets.py`, definí un nuevo `LaserPreset`.
2. Agregá a `ALL_PRESETS`.
3. Si el preset corresponde a una condición detectable por imagen, agregá regla en
   `recommend_preset()`.
4. Tests en `tests/test_laser_presets.py`.

## Reportar bugs

Abrí un Issue con:
- Versión (output de `python -c "import scripts.api_server"` + git SHA).
- OS + Python + CUDA si aplica.
- Imagen de input (o sintética reproducible).
- Params usados (`X-Preset-Applied`, params_json).
- Output esperado vs obtenido.

## Licencia de tus contribuciones

Al abrir un PR aceptás que tu código se distribuya bajo **GPL-3.0-or-later**, los
créditos te corresponden (agregá tu línea en `AUTHORS.md`), y nadie puede tomar el
trabajo combinado y cerrarlo bajo una licencia más restrictiva.

---

Dudas: abrí un Issue con la etiqueta `question`.
