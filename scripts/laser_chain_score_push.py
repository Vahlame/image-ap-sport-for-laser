#!/usr/bin/env python3
"""
Encadena busqueda -> refino -> re-render full-res y deja `best_fullres.png` fijo.

Opcionalmente ejecuta un barrido corto (--sweep) de preprocess y focus-threshold,
elige el menor score v2 leyendo best_report.json, y luego corre la cadena principal
con esos hiperparametros.

Ejemplo (todo en un comando):

  python scripts/laser_chain_score_push.py \\
    --input runs/references/foto_objetivo_sin_procesar.jpeg \\
    --target runs/references/target_imagr_acrylic.png \\
    --work-dir runs/_chain_best \\
    --sweep --quick-sweep

Raises:
    subprocess.CalledProcessError: si alguna etapa de laser_target_match falla.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import laser_runtime_env

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def _matcher_py() -> Path:
    return _SCRIPT_DIR / "laser_target_match.py"


def _read_best_score(out_dir: Path) -> float:
    path = out_dir / "best_report.json"
    if not path.is_file():
        return float("inf")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("score", float("inf")))


def _run_matcher(args: Sequence[str | Path]) -> None:
    cmd = [sys.executable, str(_matcher_py()), *[str(x) for x in args]]
    print(f"[CHAIN] {' '.join(cmd)}", flush=True)
    subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        check=True,
        env=laser_runtime_env.child_process_env(),
    )


def _sweep_configs(quick: bool) -> list[tuple[str, int, str]]:
    """
    Returns:
        Lista de (nombre_carpeta, focus_threshold, preprocess_mode).
    """
    if quick:
        return [
            ("sw_none_wide", 0, "none"),
            ("sw_none_86", 86, "none"),
            ("sw_sauvola", 0, "sauvola"),
            ("sw_niblack_82", 82, "niblack"),
        ]
    rows: list[tuple[str, int, str]] = []
    for pre in ("none", "sauvola", "niblack"):
        for ft in (0, 78, 84, 86, 92, 96):
            tag = f"sw_{pre}_f{ft}"
            rows.append((tag, ft, pre))
    return rows


def _run_sweep(
    *,
    inp: Path,
    tgt: Path,
    work: Path,
    quick: bool,
    n_mini: int,
    max_side_mini: int,
    score_version: str,
    workers: int,
    register: str,
) -> tuple[str, int, float]:
    """Devuelve (preprocess_mode, focus_threshold_ganador_mini, score_ganador)."""
    best_pre = "none"
    best_ft = 0
    best_sc = float("inf")
    for tag, ft, pre in _sweep_configs(quick):
        out = work / "sweep" / tag
        if out.is_dir():
            shutil.rmtree(out, ignore_errors=True)
        base = [
            "--input",
            inp,
            "--target",
            tgt,
            "--out",
            out,
            "--n",
            str(n_mini),
            "--max-side",
            str(max_side_mini),
            "--score-version",
            score_version,
            "--sampling",
            "sobol",
            "--register",
            register,
            "--workers",
            str(workers),
            "--guided-explore",
            "--preprocess-mode",
            pre,
        ]
        if ft > 0:
            base += ["--focus-threshold", str(ft)]
        _run_matcher(base)
        sc = _read_best_score(out)
        print(f"[SWEEP] {tag} score={sc:.6f} pre={pre} focus={ft}", flush=True)
        if sc < best_sc:
            best_sc = sc
            best_pre = pre
            best_ft = ft
    print(
        f"[SWEEP] ganador preprocess={best_pre} (mini focus={best_ft}, score={best_sc:.6f}). "
        "La cadena larga usa busqueda amplia salvo --stage1-focus.",
        flush=True,
    )
    return best_pre, best_ft, best_sc


def _chain(
    *,
    inp: Path,
    tgt: Path,
    work: Path,
    preprocess: str,
    focus_rank: int,
    n_search: int,
    n_refine: int,
    max_side_rank: int,
    score_version: str,
    workers: int,
    register: str,
    refine_top: int,
    top_report: int,
) -> Path:
    """
    Ejecuta stage1 (amplio), stage2 (refino), stage3 (full-res top-1).

    Returns:
        Ruta a `best_fullres.png` escrita bajo work_dir.
    """
    s1 = work / "stage1_search"
    s2 = work / "stage2_refine"
    s3 = work / "stage3_fullres"
    for p in (s1, s2, s3):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    stage1: list[str | Path] = [
        "--input",
        inp,
        "--target",
        tgt,
        "--out",
        s1,
        "--n",
        str(n_search),
        "--max-side",
        str(max_side_rank),
        "--score-version",
        score_version,
        "--sampling",
        "sobol",
        "--register",
        register,
        "--workers",
        str(workers),
        "--guided-explore",
        "--preprocess-mode",
        preprocess,
        "--top-report",
        str(top_report),
    ]
    if focus_rank > 0:
        stage1 += ["--focus-threshold", str(focus_rank)]
    _run_matcher(stage1)

    stage2: list[str | Path] = [
        "--input",
        inp,
        "--target",
        tgt,
        "--out",
        s2,
        "--n",
        str(n_refine),
        "--max-side",
        str(max_side_rank),
        "--score-version",
        score_version,
        "--sampling",
        "sobol",
        "--register",
        register,
        "--workers",
        str(workers),
        "--guided-explore",
        "--preprocess-mode",
        preprocess,
        "--refine-db",
        s1 / "match.sqlite",
        "--refine-top",
        str(refine_top),
        "--refine-best-per-algorithm",
        "--top-report",
        str(top_report),
    ]
    _run_matcher(stage2)

    stage3: list[str | Path] = [
        "--input",
        inp,
        "--target",
        tgt,
        "--out",
        s3,
        "--n",
        "1",
        "--max-side",
        "0",
        "--score-version",
        score_version,
        "--register",
        register,
        "--workers",
        "1",
        "--preprocess-mode",
        preprocess,
        "--from-db",
        s2 / "match.sqlite",
        "--from-db-top",
        "1",
        "--top-report",
        "5",
    ]
    _run_matcher(stage3)

    src = s3 / "match_0001.png"
    if not src.is_file():
        raise FileNotFoundError(f"No se genero {src}")
    dest = work / "best_fullres.png"
    shutil.copy2(src, dest)
    br = _read_best_score(s2)
    print(f"[CHAIN] mejor score (rank {max_side_rank}px) tras refino: {br:.6f}", flush=True)
    print(f"[CHAIN] full-res copiado a: {dest}", flush=True)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Barrido opcional + cadena search/refine/full-res -> best_fullres.png"
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("runs/_chain_push"))
    ap.add_argument("--score-version", choices=("v1", "v2", "v3", "v4"), default="v2")
    ap.add_argument("--register", choices=("none", "affine", "homography"), default="none")
    ap.add_argument("--workers", type=int, default=0, help="0 = todos los nucleos")
    ap.add_argument("--sweep", action="store_true", help="Mini-barrido preprocess/focus antes de la cadena")
    ap.add_argument(
        "--quick-sweep",
        action="store_true",
        help="Menos combinaciones en --sweep (4 en vez de 18)",
    )
    ap.add_argument("--n-mini", type=int, default=320, help="Candidatos por celda del barrido")
    ap.add_argument("--max-side-mini", type=int, default=400)
    ap.add_argument("--n-search", type=int, default=1100)
    ap.add_argument("--n-refine", type=int, default=700)
    ap.add_argument("--max-side-rank", type=int, default=520)
    ap.add_argument("--refine-top", type=int, default=8)
    ap.add_argument("--top-report", type=int, default=50)
    ap.add_argument(
        "--preprocess-mode",
        choices=("none", "sauvola", "niblack", "grabcut", "watershed", "chanvese", "deeplab", "unet", "sam2"),
        default="none",
    )
    ap.add_argument(
        "--stage1-focus",
        type=int,
        default=0,
        help="Si >0, stage1 usa --focus-threshold (busqueda local). Por defecto 0 = amplio.",
    )
    args = ap.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()

    work = args.work_dir
    if not work.is_absolute():
        work = (_REPO_ROOT / work).resolve()
    work.mkdir(parents=True, exist_ok=True)

    inp = args.input if args.input.is_absolute() else (_REPO_ROOT / args.input).resolve()
    tgt = args.target if args.target.is_absolute() else (_REPO_ROOT / args.target).resolve()

    w = args.workers if args.workers > 0 else max(1, os.cpu_count() or 2)

    preprocess = str(args.preprocess_mode)
    if args.sweep:
        pre, hint_ft, _sc = _run_sweep(
            inp=inp,
            tgt=tgt,
            work=work,
            quick=bool(args.quick_sweep),
            n_mini=int(args.n_mini),
            max_side_mini=int(args.max_side_mini),
            score_version=str(args.score_version),
            workers=w,
            register=str(args.register),
        )
        preprocess = pre
        if int(args.stage1_focus) == 0 and hint_ft > 0:
            print(
                f"[SWEEP] pista: el mini-run gano con focus={hint_ft}; "
                f"probar --stage1-focus {hint_ft} en otra pasada si conviene.",
                flush=True,
            )

    focus_chain = int(args.stage1_focus)

    _chain(
        inp=inp,
        tgt=tgt,
        work=work,
        preprocess=preprocess,
        focus_rank=focus_chain,
        n_search=int(args.n_search),
        n_refine=int(args.n_refine),
        max_side_rank=int(args.max_side_rank),
        score_version=str(args.score_version),
        workers=w,
        register=str(args.register),
        refine_top=int(args.refine_top),
        top_report=int(args.top_report),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
