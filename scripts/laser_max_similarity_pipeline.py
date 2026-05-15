#!/usr/bin/env python3
"""
Pipeline de maxima similitud (coste computacional alto): encadena exploracion amplia,
refinamiento desde SQLite, pinch de umbrales en capa densa y refino a mayor resolucion.

Disenado para --score-version v3 (similitud global + SSIM suavizado vs binario); v2 sigue soportado.

Etapas por defecto:
  1) Exploracion Sobol muy grande (full espacio de candidatos).
  2) Refine-db desde el mejor sqlite de la etapa 1 (vecindarios locales alrededor de anclas).
  3) Pinch: capa densa limitada al rango [thr-R, thr+R] del mejor de etapa 2.
  4) Opcional: misma cadena refine+pinch con --max-side-final > exploracion (halftone mas fino).

Con --full-preprocess corre tambien sauvola y niblack en etapa 1 y elige la mejor base para r2.

Ejemplo (dejar correr horas):
  python scripts/laser_max_similarity_pipeline.py \\
    --input runs/references/foto_objetivo_sin_procesar.jpeg \\
    --target runs/references/target_imagr_acrylic.png \\
    --out-root runs/_max_sim_run1

Mas rapido (prueba):
  python scripts/laser_max_similarity_pipeline.py --quick ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laser_runtime_env

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
_MATCH = _SCRIPT_DIR / "laser_target_match.py"


@dataclass(frozen=True)
class StageOutput:
    tag: str
    out_dir: Path
    sqlite: Path
    min_score: float | None
    seconds: float


def _sqlite_min_score(db: Path) -> float | None:
    if not db.is_file():
        return None
    try:
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT MIN(score) FROM matches").fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        return None


def _read_best_threshold(out_dir: Path) -> int | None:
    p = out_dir / "best_report.json"
    if not p.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        t = data.get("threshold")
        return int(t) if t is not None else None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _run_stage(
    *,
    repo: Path,
    python: Path,
    inp: Path,
    tgt: Path,
    out_dir: Path,
    workers: int,
    score_version: str,
    n: int,
    max_side: int,
    sampling: str,
    preprocess: str,
    search_preset: str,
    luma: str,
    register: str,
    refine_db: Path | None,
    refine_top: int,
    refine_breadth: str,
    explore_brutal: bool,
    dense_min: int | None,
    dense_max: int | None,
    tag: str,
    clean: bool,
) -> tuple[StageOutput, int]:
    if clean and out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str | Path] = [
        python,
        _MATCH,
        "--input",
        inp,
        "--target",
        tgt,
        "--out",
        out_dir,
        "--n",
        str(int(n)),
        "--max-side",
        str(int(max_side)),
        "--score-version",
        score_version,
        "--sampling",
        sampling,
        "--register",
        register,
        "--workers",
        str(int(workers)),
        "--guided-explore",
        "--preprocess-mode",
        preprocess,
        "--search-preset",
        search_preset,
        "--luma",
        luma,
        "--top-report",
        "32",
    ]
    if explore_brutal:
        cmd.append("--explore-brutal")
    if refine_db is not None:
        cmd += [
            "--refine-db",
            refine_db,
            "--refine-top",
            str(int(refine_top)),
            "--refine-best-per-algorithm",
            "--refine-breadth",
            refine_breadth,
        ]
    if dense_min is not None and dense_min > 0:
        cmd += ["--dense-threshold-min", str(int(dense_min))]
    if dense_max is not None and dense_max > 0:
        cmd += ["--dense-threshold-max", str(int(dense_max))]

    print(f"[PIPE] {' '.join(str(x) for x in cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(repo), env=laser_runtime_env.child_process_env())
    elapsed = time.perf_counter() - t0
    db = out_dir / "match.sqlite"
    sc = _sqlite_min_score(db)
    rc = int(proc.returncode)
    print(f"[PIPE] ({tag}) rc={rc} min_score={sc} t={elapsed:.1f}s -> {out_dir}", flush=True)
    return StageOutput(tag=tag, out_dir=out_dir, sqlite=db, min_score=sc, seconds=round(elapsed, 3)), rc


def _pick_best_explore(stages: list[StageOutput]) -> StageOutput:
    valid = [s for s in stages if s.min_score is not None]
    if not valid:
        raise RuntimeError("[PIPE] ninguna etapa de exploracion produjo scores validos")
    return min(valid, key=lambda s: s.min_score or 1e9)


def _infer_preprocess_from_explore_tag(tag: str) -> str:
    if "sauvola" in tag:
        return "sauvola"
    if "niblack" in tag:
        return "niblack"
    return "none"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pipeline multi-etapa para maximizar similitud (alto costo CPU/RAM/tiempo)"
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("runs/_max_similarity"))
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--workers", type=int, default=0, help="0 = todos los nucleos logicos")
    ap.add_argument("--score-version", choices=("v2", "v3", "v4"), default="v3")
    ap.add_argument("--luma", choices=("bt601", "bt709"), default="bt601")
    ap.add_argument("--register", choices=("none", "affine", "homography"), default="none")
    ap.add_argument("--search-preset", choices=("default", "acrylic"), default="default")

    ap.add_argument("--quick", action="store_true", help="n y max-side reducidos; omite hires y preprocess extra")
    ap.add_argument("--no-clean", action="store_true", help="No borrar carpetas de etapa si ya existen")

    ap.add_argument("--n-explore", type=int, default=0, help="0 = auto segun --quick")
    ap.add_argument("--n-refine", type=int, default=0)
    ap.add_argument("--n-pinch", type=int, default=0)
    ap.add_argument("--n-hires", type=int, default=0)

    ap.add_argument("--max-side-explore", type=int, default=0)
    ap.add_argument("--max-side-final", type=int, default=0, help="0 = omitir etapa hires")

    ap.add_argument("--refine-top", type=int, default=20)
    ap.add_argument("--pinch-radius", type=int, default=18)
    ap.add_argument(
        "--full-preprocess",
        action="store_true",
        help="Etapa1 tambien sauvola+niblack; el mejor MIN(score) alimenta refine",
    )
    ap.add_argument("--skip-sobol-baseline", action="store_true", help="No correr stage01 none (requiere --full-preprocess)")
    ap.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Si falla una etapa (rc!=0), abortar el pipeline",
    )
    ap.add_argument(
        "--refine-breadth",
        choices=("normal", "deep", "max"),
        default=None,
        help="Vecindario refine-db (etapas con --refine-db); default: normal con --quick, max sin quick",
    )
    ap.add_argument(
        "--explore-brutal",
        action="store_true",
        help="Pasa --explore-brutal a laser_target_match (plateau agresivo)",
    )
    ap.add_argument(
        "--no-explore-brutal",
        action="store_true",
        help="Desactiva explore-brutal aun en corridas largas",
    )
    args = ap.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()

    if args.no_explore_brutal:
        explore_brutal = False
    elif args.explore_brutal:
        explore_brutal = True
    else:
        explore_brutal = not args.quick

    refine_breadth = args.refine_breadth or ("normal" if args.quick else "max")

    if not _MATCH.is_file():
        print(f"No existe {_MATCH}", file=sys.stderr)
        return 2

    out_root = args.out_root if args.out_root.is_absolute() else (_REPO / args.out_root).resolve()
    inp = args.input if args.input.is_absolute() else (_REPO / args.input).resolve()
    tgt = args.target if args.target.is_absolute() else (_REPO / args.target).resolve()

    if not inp.is_file() or not tgt.is_file():
        print("[PIPE] faltan --input o --target", file=sys.stderr)
        return 2

    workers = int(args.workers) if args.workers > 0 else max(1, os.cpu_count() or 8)
    clean = not bool(args.no_clean)
    print(f"[PIPE] refine-breadth={refine_breadth} explore-brutal={explore_brutal}", flush=True)

    if args.quick:
        ne, nr, npn, nh = 400, 900, 1200, 2500
        msex, msfin = 320, 420
    else:
        ne, nr, npn, nh = 6500, 14000, 18000, 12000
        msex, msfin = 560, 720

    n_explore = args.n_explore or ne
    n_refine = args.n_refine or nr
    n_pinch = args.n_pinch or npn
    n_hires = args.n_hires or nh
    max_side_explore = args.max_side_explore or msex
    max_side_final = args.max_side_final if args.max_side_final > 0 else (0 if args.quick else msfin)

    explore_outputs: list[StageOutput] = []
    rc_any = 0

    def run(tag: str, **kw: Any) -> StageOutput:
        nonlocal rc_any
        st, rc = _run_stage(tag=tag, **kw)
        if rc != 0:
            rc_any = rc
            if args.fail_fast:
                raise RuntimeError(f"[PIPE] etapa {tag} fallo rc={rc}")
        return st

    if args.skip_sobol_baseline and not args.full_preprocess:
        print("[PIPE] error: --skip-sobol-baseline sin --full-preprocess no tiene exploracion", file=sys.stderr)
        return 2

    if not args.skip_sobol_baseline:
        explore_outputs.append(
            run(
                repo=_REPO,
                python=args.python,
                inp=inp,
                tgt=tgt,
                out_dir=out_root / "stage01_explore_sobol_none",
                workers=workers,
                score_version=args.score_version,
                n=n_explore,
                max_side=max_side_explore,
                sampling="sobol",
                preprocess="none",
                search_preset=args.search_preset,
                luma=args.luma,
                register=args.register,
                refine_db=None,
                refine_top=args.refine_top,
                refine_breadth=refine_breadth,
                explore_brutal=explore_brutal,
                dense_min=None,
                dense_max=None,
                tag="stage01_explore_sobol_none",
                clean=clean,
            )
        )

    pre_n = max(200, n_explore // 3) if args.quick else max(1200, n_explore // 2)
    if args.full_preprocess:
        for pre, tag in (("sauvola", "stage01_explore_sobol_sauvola"), ("niblack", "stage01_explore_sobol_niblack")):
            explore_outputs.append(
                run(
                    repo=_REPO,
                    python=args.python,
                    inp=inp,
                    tgt=tgt,
                    out_dir=out_root / tag,
                    workers=workers,
                    score_version=args.score_version,
                    n=pre_n,
                    max_side=max_side_explore,
                    sampling="sobol",
                    preprocess=pre,
                    search_preset=args.search_preset,
                    luma=args.luma,
                    register=args.register,
                    refine_db=None,
                    refine_top=args.refine_top,
                    refine_breadth=refine_breadth,
                    explore_brutal=explore_brutal,
                    dense_min=None,
                    dense_max=None,
                    tag=tag,
                    clean=clean,
                )
            )

    if not explore_outputs:
        print("[PIPE] sin etapas de exploracion", file=sys.stderr)
        return 2

    best_ex = _pick_best_explore(explore_outputs)
    print(
        f"[PIPE] mejor exploracion: {best_ex.tag} min_score={best_ex.min_score:.8f} sqlite={best_ex.sqlite}",
        flush=True,
    )

    pre_mode = _infer_preprocess_from_explore_tag(best_ex.tag)

    try:
        s2 = run(
            repo=_REPO,
            python=args.python,
            inp=inp,
            tgt=tgt,
            out_dir=out_root / "stage02_refine",
            workers=workers,
            score_version=args.score_version,
            n=n_refine,
            max_side=max_side_explore,
            sampling="grid",
            preprocess=pre_mode,
            search_preset=args.search_preset,
            luma=args.luma,
            register=args.register,
            refine_db=best_ex.sqlite,
            refine_top=args.refine_top,
            refine_breadth=refine_breadth,
            explore_brutal=explore_brutal,
            dense_min=None,
            dense_max=None,
            tag="stage02_refine",
            clean=clean,
        )

        th = _read_best_threshold(s2.out_dir)
        lo: int | None = None
        hi: int | None = None
        if th is not None:
            lo = max(1, th - int(args.pinch_radius))
            hi = min(254, th + int(args.pinch_radius))
            print(f"[PIPE] pinch threshold {th} -> [{lo},{hi}]", flush=True)

        s3 = run(
            repo=_REPO,
            python=args.python,
            inp=inp,
            tgt=tgt,
            out_dir=out_root / "stage03_pinch",
            workers=workers,
            score_version=args.score_version,
            n=n_pinch,
            max_side=max_side_explore,
            sampling="sobol",
            preprocess=pre_mode,
            search_preset=args.search_preset,
            luma=args.luma,
            register=args.register,
            refine_db=s2.sqlite,
            refine_top=args.refine_top,
            refine_breadth=refine_breadth,
            explore_brutal=explore_brutal,
            dense_min=lo,
            dense_max=hi,
            tag="stage03_pinch",
            clean=clean,
        )

        stages_order: list[StageOutput] = [*explore_outputs, s2, s3]
        valid_for_best = [s for s in stages_order if s.min_score is not None]
        best_global_stage = min(valid_for_best, key=lambda s: s.min_score or 1e9) if valid_for_best else None
        global_best = best_global_stage.min_score if best_global_stage else None
        best_dir = (best_global_stage.out_dir if best_global_stage else s3.out_dir)
        best_stage = (best_global_stage.tag if best_global_stage else "stage03_pinch")

        hires_out: StageOutput | None = None
        if max_side_final > max_side_explore and not args.quick:
            hires_out = run(
                repo=_REPO,
                python=args.python,
                inp=inp,
                tgt=tgt,
                out_dir=out_root / "stage04_hires_refine",
                workers=workers,
                score_version=args.score_version,
                n=n_hires,
                max_side=max_side_final,
                sampling="grid",
                preprocess=pre_mode,
                search_preset=args.search_preset,
                luma=args.luma,
                register=args.register,
                refine_db=s3.sqlite,
                refine_top=max(args.refine_top, 20),
                refine_breadth=refine_breadth,
                explore_brutal=explore_brutal,
                dense_min=None,
                dense_max=None,
                tag="stage04_hires_refine",
                clean=clean,
            )
            stages_order.append(hires_out)
            if hires_out.min_score is not None:
                if global_best is None or hires_out.min_score < global_best:
                    global_best = hires_out.min_score
                    best_dir = hires_out.out_dir
                    best_stage = hires_out.tag

        summary = {
            "score_version": args.score_version,
            "refine_breadth": refine_breadth,
            "explore_brutal": explore_brutal,
            "input": str(inp),
            "target": str(tgt),
            "out_root": str(out_root),
            "workers": workers,
            "global_best_min_score": global_best,
            "best_stage": best_stage,
            "best_out_dir": str(best_dir),
            "pinch_threshold": th,
            "pinch_range": [lo, hi] if lo is not None else None,
            "explore_winner": best_ex.tag,
            "preprocess_used_after_explore": pre_mode,
            "stages": [
                {
                    "tag": s.tag,
                    "out": str(s.out_dir),
                    "min_score": s.min_score,
                    "seconds": s.seconds,
                }
                for s in stages_order
            ],
        }
        (out_root / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[PIPE] resumen -> {out_root / 'pipeline_summary.json'}", flush=True)
        print(
            f"[PIPE] MEJOR GLOBAL score={global_best} etapa={best_stage} dir={best_dir}",
            flush=True,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0 if rc_any == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
