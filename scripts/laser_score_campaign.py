#!/usr/bin/env python3
"""
Campana de muchas corridas cortas de laser_target_match para comparar scores v2/v3.

Recorre preprocess, focus-threshold, muestreo; ronda 2 refina desde match.sqlite de cada
celda top-k (mismo max-side). Ronda 3 opcional: un refino extra con --dense-threshold-min/max
centrado en el umbral del mejor de r2 (afinacion tipo pinch).

Ejemplo:

  python scripts/laser_score_campaign.py \\
    --input runs/references/foto_objetivo_sin_procesar.jpeg \\
    --target runs/references/target_imagr_acrylic.png \\
    --out-root runs/_campaign_A --rounds 3 --aggressive
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import laser_runtime_env

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MATCH = _SCRIPT_DIR / "laser_target_match.py"


@dataclass(frozen=True)
class Cell:
    tag: str
    preprocess: str
    focus: int
    sampling: str
    register: str
    n: int
    max_side: int
    refine_db: Path | None = None
    refine_top: int = 8
    dense_threshold_min: int | None = None
    dense_threshold_max: int | None = None


def _read_score(out_dir: Path) -> float | None:
    p = out_dir / "best_report.json"
    if not p.is_file():
        return None
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return float(data["score"])


def _read_best_threshold(out_dir: Path) -> int | None:
    p = out_dir / "best_report.json"
    if not p.is_file():
        return None
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    t = data.get("threshold")
    if t is None:
        return None
    return int(t)


def _run_cell(
    inp: Path,
    tgt: Path,
    cell: Cell,
    out_root: Path,
    workers: int,
    score_version: str,
) -> tuple[Cell, Path, float | None, int]:
    out = out_root / cell.tag
    if out.is_dir():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    cmd: list[str] = [
        sys.executable,
        str(_MATCH),
        "--input",
        str(inp),
        "--target",
        str(tgt),
        "--out",
        str(out),
        "--n",
        str(cell.n),
        "--max-side",
        str(cell.max_side),
        "--score-version",
        score_version,
        "--sampling",
        cell.sampling,
        "--register",
        cell.register,
        "--workers",
        str(workers),
        "--guided-explore",
        "--preprocess-mode",
        cell.preprocess,
        "--top-report",
        "24",
    ]
    if cell.focus > 0:
        cmd += ["--focus-threshold", str(cell.focus)]
    if cell.refine_db is not None:
        cmd += [
            "--refine-db",
            str(cell.refine_db),
            "--refine-top",
            str(int(cell.refine_top)),
            "--refine-best-per-algorithm",
        ]
    if cell.dense_threshold_min is not None and cell.dense_threshold_min > 0:
        cmd += ["--dense-threshold-min", str(int(cell.dense_threshold_min))]
    if cell.dense_threshold_max is not None and cell.dense_threshold_max > 0:
        cmd += ["--dense-threshold-max", str(int(cell.dense_threshold_max))]
    print(f"[CAMP] {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=laser_runtime_env.child_process_env())
    sc = _read_score(out)
    return cell, out, sc, int(rc.returncode)


def _grid_round1(quick: bool, aggressive: bool) -> list[Cell]:
    pre = ("none", "sauvola", "niblack")
    if quick:
        focuses = (0, 78, 86, 92)
        samplings = ("sobol",)
        registers = ("none",)
        n, ms = 240, 340
    else:
        focuses = (0, 72, 78, 84, 86, 92, 98)
        samplings = ("sobol", "grid")
        registers = ("none",)
        n, ms = 320, 400
    cells: list[Cell] = []
    for pre_m, foc, samp, reg in itertools.product(pre, focuses, samplings, registers):
        tag = f"r1_{pre_m}_f{foc}_{samp}_{reg}".replace(".", "p")
        cells.append(
            Cell(
                tag=tag,
                preprocess=pre_m,
                focus=int(foc),
                sampling=samp,
                register=reg,
                n=n,
                max_side=ms,
            )
        )
    if aggressive:
        na, ma = (200, 320) if quick else (260, 380)
        for pre_m in ("none", "sauvola", "niblack"):
            cells.append(
                Cell(
                    tag=f"r1ag_{pre_m}_grid_f0",
                    preprocess=pre_m,
                    focus=0,
                    sampling="grid",
                    register="none",
                    n=na,
                    max_side=ma,
                )
            )
        cells.append(
            Cell(
                tag="r1ag_none_pinch707090",
                preprocess="none",
                focus=0,
                sampling="sobol",
                register="none",
                n=na,
                max_side=ma,
                dense_threshold_min=70,
                dense_threshold_max=90,
            )
        )
    return cells


def _grid_round2(
    entries: Sequence[tuple[Cell, Path, float]],
    quick: bool,
    refine_top: int,
) -> list[Cell]:
    n2 = 650 if quick else 1100
    cells: list[Cell] = []
    for c0, p0, _sc0 in entries:
        db = p0 / "match.sqlite"
        if not db.is_file():
            print(f"[CAMP] r2 skip sin sqlite: {db}", flush=True)
            continue
        tag = f"r2_{c0.tag}"
        cells.append(
            Cell(
                tag=tag,
                preprocess=c0.preprocess,
                focus=c0.focus,
                sampling=c0.sampling,
                register="none",
                n=n2,
                max_side=int(c0.max_side),
                refine_db=db,
                refine_top=int(refine_top),
            )
        )
    return cells


def _round3_pinch_cell(
    best_r2_dir: Path,
    preprocess: str,
    focus: int,
    sampling: str,
    n3: int,
    max_side: int,
    pinch_radius: int,
    refine_top: int,
) -> Cell | None:
    db = best_r2_dir / "match.sqlite"
    if not db.is_file():
        return None
    th = _read_best_threshold(best_r2_dir)
    if th is None:
        print("[CAMP] r3: sin threshold en best_report; omitiendo pinch", flush=True)
        return Cell(
            tag="r3_refine_only",
            preprocess=preprocess,
            focus=focus,
            sampling=sampling,
            register="none",
            n=n3,
            max_side=max_side,
            refine_db=db,
            refine_top=refine_top,
        )
    lo = max(1, th - pinch_radius)
    hi = min(254, th + pinch_radius)
    print(f"[CAMP] r3 pinch alrededor de threshold={th} -> [{lo},{hi}]", flush=True)
    return Cell(
        tag=f"r3_pinch_thr{th}",
        preprocess=preprocess,
        focus=focus,
        sampling=sampling,
        register="none",
        n=n3,
        max_side=max_side,
        refine_db=db,
        refine_top=refine_top,
        dense_threshold_min=lo,
        dense_threshold_max=hi,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Campana multi-celda para minimizar score v2")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("runs/_score_campaign"))
    ap.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="1=r1 sola; 2=r1+r2 refine; 3=r1+r2+r3 pinch sobre mejor r2",
    )
    ap.add_argument("--top-k", type=int, default=6, help="Cuantas celdas repetir en ronda 2")
    ap.add_argument("--quick", action="store_true", help="Menos celdas y n mas chico")
    ap.add_argument("--aggressive", action="store_true", help="Celdas extra: grid + pinch 70-90 en none")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--score-version", choices=("v1", "v2", "v3", "v4"), default="v2")
    ap.add_argument("--refine-top", type=int, default=8, help="Anclas refine-db (r2/r3)")
    ap.add_argument("--r3-n", type=int, default=900, help="Candidatos en ronda 3 (pinch)")
    ap.add_argument("--r3-pinch-radius", type=int, default=12, help="Radio +/- en umbral para capa densa")
    args = ap.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()

    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = (_REPO_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    inp = args.input if args.input.is_absolute() else (_REPO_ROOT / args.input).resolve()
    tgt = args.target if args.target.is_absolute() else (_REPO_ROOT / args.target).resolve()

    w = int(args.workers) if args.workers > 0 else max(1, os.cpu_count() or 2)

    results: list[tuple[Cell, Path, float | None]] = []
    for cell in _grid_round1(bool(args.quick), bool(args.aggressive)):
        c, outp, sc, rc = _run_cell(inp, tgt, cell, out_root, w, str(args.score_version))
        if rc != 0:
            print(f"[CAMP] FAIL rc={rc} tag={cell.tag}", flush=True)
        print(
            f"[CAMP] score={sc if sc is not None else 'n/a'} "
            f"pre={c.preprocess} focus={c.focus} samp={c.sampling} reg={c.register} -> {outp}",
            flush=True,
        )
        results.append((c, outp, sc))

    ranked = [(c, p, s) for c, p, s in results if s is not None]
    ranked.sort(key=lambda t: t[2])
    summary = {
        "input": str(inp),
        "target": str(tgt),
        "round": 1,
        "best": [
            {
                "score": s,
                "preprocess": c.preprocess,
                "focus": c.focus,
                "sampling": c.sampling,
                "register": c.register,
                "out": str(p),
            }
            for c, p, s in ranked[:40]
        ],
    }
    (out_root / "campaign_summary_r1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not ranked:
        print("[CAMP] sin scores validos", flush=True)
        return 2

    best_c, best_p, best_s = ranked[0]
    print(
        f"[CAMP] MEJOR r1: score={best_s:.6f} pre={best_c.preprocess} focus={best_c.focus} "
        f"samp={best_c.sampling} reg={best_c.register} dir={best_p}",
        flush=True,
    )

    global_best = ranked[0][2]
    best_dir = ranked[0][1]
    best_round = "r1"
    r2_ok: list[tuple[Cell, Path, float]] = []

    if int(args.rounds) >= 2 and ranked:
        top = ranked[: int(args.top_k)]
        r2_cells = _grid_round2(top, bool(args.quick), int(args.refine_top))
        r2_results: list[tuple[Cell, Path, float | None]] = []
        for cell in r2_cells:
            c, outp, sc, rc = _run_cell(inp, tgt, cell, out_root, w, str(args.score_version))
            if rc != 0:
                print(f"[CAMP] r2 FAIL rc={rc} tag={cell.tag}", flush=True)
            print(f"[CAMP] r2 score={sc} tag={cell.tag}", flush=True)
            r2_results.append((c, outp, sc))
        r2_ok = [(c, p, s) for c, p, s in r2_results if s is not None]
        r2_ok.sort(key=lambda t: t[2])
        summary2 = {
            "round": 2,
            "note": "refine-db desde celda r1; mismo max-side que r1",
            "best": [
                {
                    "score": s,
                    "preprocess": c.preprocess,
                    "focus": c.focus,
                    "sampling": c.sampling,
                    "register": c.register,
                    "out": str(p),
                }
                for c, p, s in r2_ok[:25]
            ],
        }
        (out_root / "campaign_summary_r2.json").write_text(
            json.dumps(summary2, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if r2_ok:
            b2, p2, s2 = r2_ok[0]
            print(
                f"[CAMP] MEJOR r2 (top celda): score={s2:.6f} pre={b2.preprocess} focus={b2.focus} "
                f"samp={b2.sampling} dir={p2}",
                flush=True,
            )
            for _c, p, s in r2_ok:
                if s < global_best:
                    global_best = s
                    best_dir = p
                    best_round = "r2"

    if int(args.rounds) >= 3 and r2_ok:
        b2, p2, s2 = r2_ok[0]
        ms = int(b2.max_side)
        r3_cell = _round3_pinch_cell(
            p2,
            b2.preprocess,
            b2.focus,
            b2.sampling,
            int(args.r3_n),
            ms,
            int(args.r3_pinch_radius),
            int(args.refine_top),
        )
        if r3_cell is not None:
            c3, p3, s3, rc3 = _run_cell(inp, tgt, r3_cell, out_root, w, str(args.score_version))
            if rc3 != 0:
                print(f"[CAMP] r3 FAIL rc={rc3}", flush=True)
            print(f"[CAMP] r3 score={s3} tag={r3_cell.tag}", flush=True)
            if s3 is not None:
                summary3 = {
                    "round": 3,
                    "note": "refine-db desde mejor r2 + pinch capa densa en umbral del best_report",
                    "score": s3,
                    "out": str(p3),
                }
                (out_root / "campaign_summary_r3.json").write_text(
                    json.dumps(summary3, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                if s3 < global_best:
                    global_best = s3
                    best_dir = p3
                    best_round = "r3"
                print(f"[CAMP] MEJOR tras r3: score={s3:.6f} dir={p3}", flush=True)

    print(f"[CAMP] resumen final: score={global_best:.6f} dir={best_dir} ronda={best_round}", flush=True)

    best_json = {
        "best_score": global_best,
        "best_out_dir": str(best_dir),
        "best_round": best_round,
        "out_root": str(out_root),
    }
    (out_root / "campaign_best.json").write_text(
        json.dumps(best_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[CAMP] campaign_best.json -> {out_root / 'campaign_best.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
