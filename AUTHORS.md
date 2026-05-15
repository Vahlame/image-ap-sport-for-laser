# AUTHORS

Image AP — Laser Image Prep es un proyecto comunitario. Esta lista reconoce a
contribuyentes en orden cronológico de primer aporte. Para añadirte, mandá un PR
agregando tu línea (formato: `Nombre o handle <email opcional> — qué aportaste`).

## Mantenedor inicial

- **Vahlame** (`https://github.com/Vahlame`) — diseño del proyecto, integración del
  workflow operador, validación física en Funsun 50W, dirección general.

## Contribuciones técnicas asistidas

- **Claude (Anthropic) — agentes Cursor / Claude Code 2026-05** — implementación del
  motor de halftone (40+ algoritmos), física CO2 cableada (DPI cap, LUT material,
  scaled USM), scoring v1..v5, calibration workflow (step-wedge + fit LUT),
  simulador de grabado físico, API FastAPI, wizard SvelteKit, sistema de presets +
  auto-detector, ~120 tests automatizados, documentación.

## Referencias / Inspiración

Los algoritmos de halftone son dominio público; las implementaciones son propias
basadas en los papers originales. Reconocimientos a los autores históricos:

- Robert W. Floyd & Louis Steinberg (1976) — error diffusion 4-vecinos.
- J.F. Jarvis, C.N. Judice, W.H. Ninke (1976) — kernel 12-vecinos.
- Peter Stucki (1981) — kernel mejorado.
- Bill Atkinson (1986) — diffusion fraccional Atkinson.
- Daniel Burkes (1988) — kernel 7-vecinos.
- Frankie Sierra (1989) — kernels Sierra 3/2/Lite.
- Robert Ulichney (1993) — método void-and-cluster para blue-noise.
- Zhou Wang et al. (2003) — SSIM.
- Anthropic Research — LPIPS (vía paquete `lpips`).

Vendors / dependencias:
- Lorem Picsum / Unsplash — imágenes de stock para tests automatizados.
- Wikimedia Commons / NASA — imágenes PD para tests adicionales.

---

*Si tu nombre falta y aportaste algo, abrí un PR.*
