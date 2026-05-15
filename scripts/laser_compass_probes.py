#!/usr/bin/env python3
"""
Pruebas cortas inspiradas en flujo acrilico / halftone (Compass): BT.709 vs BT.601 en luma,
re-evaluacion desde SQLite de campana si existe, y mini-grid comparable.

Uso:
  python scripts/laser_compass_probes.py
  python scripts/laser_compass_probes.py --workers 8 --from-db runs/_campaign_push/r3_pinch_thr80/match.sqlite
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import laser_runtime_env

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MATCH_SCRIPT = _SCRIPT_DIR / "laser_target_match.py"

_SCORE_RE = re.compile(r"Mejor:\s+.*score=([\d.]+)", re.MULTILINE)


def _parse_best_score(stdout: str) -> float | None:
    m = _SCORE_RE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


def _run_probe(
    label: str,
    out_dir: Path,
    py: Path,
    extra: list[str],
) -> tuple[str, float | None, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        str(_MATCH_SCRIPT),
        *extra,
        "--out",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=laser_runtime_env.child_process_env(),
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    score = _parse_best_score(proc.stdout or "")
    if score is None:
        score = _parse_best_score(text)
    return label, score, int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probes BT.601 vs BT.709 y re-eval desde DB (Compass)")
    ap.add_argument("--python", type=Path, default=Path(sys.executable), help="Interprete Python")
    ap.add_argument("--input", type=Path, default=_REPO_ROOT / "runs/references/foto_objetivo_sin_procesar.jpeg")
    ap.add_argument("--target", type=Path, default=_REPO_ROOT / "runs/references/target_imagr_acrylic.png")
    ap.add_argument(
        "--from-db",
        type=Path,
        default=_REPO_ROOT / "runs/_campaign_push/r3_pinch_thr80/match.sqlite",
        help="SQLite de campana (si no existe se omiten pruebas from-db)",
    )
    ap.add_argument("--workers", type=int, default=min(8, __import__("os").cpu_count() or 4))
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "runs/_compass_probes")
    args = ap.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()

    if not _MATCH_SCRIPT.is_file():
        print(f"No se encuentra {_MATCH_SCRIPT}", file=sys.stderr)
        return 2
    if not args.input.is_file() or not args.target.is_file():
        print("Faltan --input o --target por defecto en runs/references/.", file=sys.stderr)
        return 2

    base_common = [
        "--input",
        str(args.input),
        "--target",
        str(args.target),
        "--score-version",
        "v2",
        "--preprocess-mode",
        "none",
        "--register",
        "none",
        "--workers",
        str(args.workers),
        "--guided-explore",
        "--top-report",
        "8",
    ]

    rows: list[tuple[str, float | None, int]] = []
    out_root = args.out_root

    if args.from_db.is_file():
        for luma, tag in (("bt601", "fromdb_bt601"), ("bt709", "fromdb_bt709")):
            label, score, rc = _run_probe(
                f"from-db top16 luma={luma} max-side=340",
                out_root / tag,
                args.python,
                base_common
                + [
                    "--luma",
                    luma,
                    "--from-db",
                    str(args.from_db),
                    "--from-db-top",
                    "16",
                    "--max-side",
                    "340",
                    "--n",
                    "24",
                ],
            )
            rows.append((label, score, rc))
    else:
        print(f"[SKIP] from-db: no existe {args.from_db}", flush=True)

    for luma, tag in (("bt601", "grid_bt601"), ("bt709", "grid_bt709")):
        label, score, rc = _run_probe(
            f"grid n=96 luma={luma} max-side=320",
            out_root / tag,
            args.python,
            base_common
            + [
                "--luma",
                luma,
                "--sampling",
                "grid",
                "--n",
                "96",
                "--max-side",
                "320",
            ],
        )
        rows.append((label, score, rc))

    print("\n=== laser_compass_probes (menor score = mejor en v2) ===\n")
    w = max(len(r[0]) for r in rows) if rows else 40
    for label, score, rc in rows:
        sc = f"{score:.6f}" if score is not None else "n/a"
        print(f"{label:{w}}  score={sc}  rc={rc}")
    if any(rc != 0 for _, _, rc in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
