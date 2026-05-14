"""CLI: sugerencia heuristica de comando laser_target_match basada en historial."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Propuesta de CLI desde historial")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()
    inp = args.input.expanduser()
    tgt = args.target.expanduser()
    if not inp.is_absolute():
        cand = args.repo / inp
        inp = cand.resolve() if cand.is_file() else inp.resolve()
    else:
        inp = inp.resolve()
    if not tgt.is_absolute():
        cand_t = args.repo / tgt
        tgt = cand_t.resolve() if cand_t.is_file() else tgt.resolve()
    else:
        tgt = tgt.resolve()

    ih = _sha256_file(inp)
    th = _sha256_file(tgt)

    dbp = args.repo / "runs" / "_meta" / "history.sqlite"
    if not dbp.is_file():
        print("[META] sin historial; sugerencia generica:")
        print(
            f"  python scripts/laser_target_match.py --input {args.input} --target {args.target} "
            f"--out runs/proposed_manual --n 4000 --score-version v2 --sampling sobol --guided-explore"
        )
        return 0
    con = sqlite3.connect(str(dbp))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT best_params_json, score_version, preprocess_mode
        FROM experiments
        WHERE input_hash = ? OR target_hash = ?
        ORDER BY id DESC LIMIT 30
        """,
        (ih, th),
    ).fetchall()
    if not rows:
        rows = con.execute(
            "SELECT best_params_json, score_version, preprocess_mode FROM experiments ORDER BY id DESC LIMIT 30"
        ).fetchall()
    con.close()
    thr_vals: list[int] = []
    for r in rows:
        try:
            d = json.loads(r["best_params_json"] or "{}")
            if "threshold" in d:
                thr_vals.append(int(d["threshold"]))
        except (TypeError, ValueError, KeyError):
            continue
    if thr_vals:
        lo, hi = min(thr_vals), max(thr_vals)
        ft = (lo + hi) // 2
    else:
        lo, hi, ft = 0, 0, 96
    print("Recomendacion basada en historial reciente (heuristica):")
    print(
        f"  python scripts/laser_target_match.py --input {args.input} --target {args.target} "
        f"--out runs/proposed_from_meta --n 6000 --score-version v2 --sampling sobol "
        f"--register affine --guided-explore --focus-threshold {ft}"
    )
    if thr_vals:
        print(f"  Razon: ventana empirica de threshold en runs similares aprox [{lo}, {hi}]")
    else:
        print("  Razon: pocos datos; umbral por defecto centrado en 96")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
