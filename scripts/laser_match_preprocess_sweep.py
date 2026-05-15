#!/usr/bin/env python3
"""
Reparte --total-n evaluaciones entre varios --preprocess-mode (una carpeta por modo).

Cada subcorrida usa la misma rejilla de algoritmos que laser_target_match.py (según su --n).
U-Net se omite si no pasas --unet-weights.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import laser_runtime_env


def split_counts(total: int, n_modes: int) -> list[int]:
    base = total // n_modes
    rem = total % n_modes
    return [base + (1 if i < rem else 0) for i in range(n_modes)]


def main() -> int:
    p = argparse.ArgumentParser(description="Barrido de preprocess-mode con reparto de --n")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Carpeta base; se crean subcarpetas pre_<modo>/")
    p.add_argument("--total-n", type=int, default=10_000, help="Suma de candidatos entre todos los modos")
    p.add_argument("--max-side", type=int, default=520)
    p.add_argument("--top-report", type=int, default=80)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--sam2-prompts",
        type=Path,
        default=Path("runs/references/sam2_prompt_example.json"),
        help="JSON SAM2 (solo se usa en modo sam2)",
    )
    p.add_argument("--unet-weights", type=Path, default=None, help="Si se indica, se incluye preprocess unet")
    p.add_argument(
        "--exclude-modes",
        type=str,
        default="",
        help="Lista separada por comas (p.ej. deeplab,sam2) para omitir modos.",
    )
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Si un modo falla (deps), registrar y seguir con el siguiente.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()

    repo = Path(__file__).resolve().parents[1]
    match_py = repo / "scripts" / "laser_target_match.py"

    modes: list[str] = [
        "none",
        "sauvola",
        "niblack",
        "grabcut",
        "watershed",
        "chanvese",
        "deeplab",
        "sam2",
    ]
    if args.unet_weights is not None:
        modes.append("unet")

    skip = {x.strip().lower() for x in args.exclude_modes.split(",") if x.strip()}
    modes = [m for m in modes if m not in skip]
    if not modes:
        print("No quedan modos tras --exclude-modes", file=sys.stderr)
        return 2

    counts = split_counts(int(args.total_n), len(modes))
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = args.out / "sweep_preprocess_manifest.txt"
    lines: list[str] = []
    failures = 0

    for mode, n in zip(modes, counts, strict=True):
        if n <= 0:
            continue
        sub = args.out / f"pre_{mode}"
        cmd = [
            sys.executable,
            str(match_py),
            "--input",
            str(args.input.resolve()),
            "--target",
            str(args.target.resolve()),
            "--out",
            str(sub.resolve()),
            "--preprocess-mode",
            mode,
            "--n",
            str(n),
            "--max-side",
            str(args.max_side),
            "--top-report",
            str(args.top_report),
            "--workers",
            str(args.workers),
        ]
        if mode == "sam2":
            cmd.extend(["--sam2-prompts", str((repo / args.sam2_prompts).resolve() if not args.sam2_prompts.is_absolute() else args.sam2_prompts)])
        if mode == "chanvese":
            cmd.append("--no-chanvese-log-progress")
        if mode == "unet":
            cmd.extend(["--unet-weights", str(args.unet_weights.resolve())])

        line = f"{mode}\tn={n}\t{sub}"
        print(line, flush=True)
        lines.append(line)

        if args.dry_run:
            continue

        sub.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, cwd=str(repo), env=laser_runtime_env.child_process_env())
        if r.returncode != 0:
            err = f"[ERROR] modo={mode} exit={r.returncode}"
            print(err, file=sys.stderr)
            lines.append(err)
            failures += 1
            if not args.continue_on_error:
                manifest.write_text("\n".join(lines) + f"\nFAILED_AT={mode}\n", encoding="utf-8")
                return r.returncode
            continue

    tail = "OK\n" if failures == 0 else f"PARTIAL failures={failures}\n"
    manifest.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")
    print(f"Manifest: {manifest}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
