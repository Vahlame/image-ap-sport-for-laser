#!/usr/bin/env python3
"""
Meta-optimizador Optuna (TPE) sobre el mismo render+score que laser_target_match.

Requiere extra: pip install -e ".[meta]"
Persistencia: sqlite en out_dir/optuna.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None  # type: ignore[assignment]

import laser_scoring
import laser_target_match as ltm


OPTUNA_ALGORITHMS = (
    "floyd",
    "floyd_serpentine",
    "burkes",
    "burkes_serpentine",
    "sierra3",
    "bayer8",
    "bayer4",
)


def _prepare_grays(
    input_path: Path,
    target_path: Path,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Carga RGB, reescala como main (sin preprocess ni ECC)."""
    inp = ltm.load_rgb(input_path)
    tgt_full = ltm.load_rgb(target_path)
    tgt = tgt_full.copy()
    if max_side > 0:
        tgt.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    inp_r = ltm.resize_to_target(inp, tgt.size, max_side=0)
    rgb_u8 = np.array(inp_r)
    base_gray = ltm.rgb_to_gray(rgb_u8)
    target_gray = ltm.rgb_to_gray(np.array(tgt))
    if ltm.detect_binary_target(target_gray):
        target_binary = np.where(target_gray > 127, 255, 0).astype(np.uint8)
    else:
        tt = ltm.otsu_threshold(target_gray)
        target_binary = np.where(target_gray >= tt, 255, 0).astype(np.uint8)
    target_density = ltm.density_map(target_gray)
    target_edges = ltm.edge_map(target_gray)
    tw = float(np.mean(target_binary == 255))
    return base_gray, target_gray, target_binary, target_density, target_edges, tw


def main() -> int:
    if optuna is None:
        print("[OPTUNA] paquete no instalado; instalar con: pip install -e \".[meta]\"", file=sys.stderr)
        return 0

    ap = argparse.ArgumentParser(description="Optuna TPE sobre candidatos laser")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max-side", type=int, default=240)
    ap.add_argument("--score-version", choices=("v1", "v2", "v3", "v4"), default="v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(args.out / 'optuna.db').as_posix()}"

    base_gray, target_gray, target_binary, target_density, target_edges, _ = _prepare_grays(
        args.input,
        args.target,
        args.max_side,
    )

    def objective(trial: Any) -> float:
        cand = ltm.Candidate(
            algorithm=trial.suggest_categorical("algorithm", OPTUNA_ALGORITHMS),
            invert=bool(trial.suggest_int("invert", 0, 1)),
            threshold=int(trial.suggest_int("threshold", 1, 254)),
            contrast=float(trial.suggest_float("contrast", 0.35, 1.0)),
            brightness=float(trial.suggest_float("brightness", -40.0, 40.0)),
            gamma=float(trial.suggest_float("gamma", 0.75, 1.15)),
            autocontrast=float(trial.suggest_float("autocontrast", 0.0, 4.0)),
            sharpen=float(trial.suggest_float("sharpen", 0.0, 120.0)),
        )
        out = ltm.render_candidate(base_gray, cand)
        score, _, _, _ = laser_scoring.score_candidate_dispatch(
            str(args.score_version),
            out,
            target_gray,
            target_binary,
            target_density,
            target_edges,
            cand,
        )
        return float(score)

    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name="laser_match",
        load_if_exists=True,
        sampler=TPESampler(seed=int(args.seed)),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)

    topk = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))[:40]
    export: list[dict[str, Any]] = []
    seed_db = args.out / "optuna_match_seed.sqlite"
    if seed_db.exists():
        seed_db.unlink()
    conn = ltm.ensure_db(seed_db)
    for tr in topk:
        if not tr.params or tr.value is None:
            continue
        p = tr.params
        cand = ltm.Candidate(
            algorithm=str(p["algorithm"]),
            invert=bool(p["invert"]),
            threshold=int(p["threshold"]),
            contrast=float(p["contrast"]),
            brightness=float(p["brightness"]),
            gamma=float(p["gamma"]),
            autocontrast=float(p["autocontrast"]),
            sharpen=float(p["sharpen"]),
        )
        out = ltm.render_candidate(base_gray, cand)
        score, px, ee, wr = laser_scoring.score_candidate_dispatch(
            str(args.score_version),
            out,
            target_gray,
            target_binary,
            target_density,
            target_edges,
            cand,
        )
        export.append({"value": score, "params": p})
        conn.execute(
            """
            INSERT INTO matches (
                algorithm, invert, threshold, contrast, brightness, gamma, autocontrast, sharpen,
                score, pixel_error, edge_error, white_ratio, target_white_ratio, output_file, seconds, created
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            """,
            (
                cand.algorithm,
                int(cand.invert),
                cand.threshold,
                cand.contrast,
                cand.brightness,
                cand.gamma,
                cand.autocontrast,
                cand.sharpen,
                score,
                px,
                ee,
                wr,
                float(np.mean(target_binary == 255)),
                "optuna_seed.png",
                0.0,
            ),
        )
    conn.commit()
    conn.close()

    (args.out / "optuna_top.json").write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"[OPTUNA] best_value={study.best_value:.6f} trials={len(study.trials)} storage={storage}", flush=True)
    print(
        "Refinar con grid guiado:\n"
        f"  python scripts/laser_target_match.py --input {args.input} --target {args.target} "
        f"--out runs/from_optuna_refine --n 2000 --score-version {args.score_version} "
        f"--from-db {seed_db} --from-db-top 40"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
