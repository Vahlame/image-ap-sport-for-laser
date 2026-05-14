"""CLI: resumen de experimentos en runs/_meta/history.sqlite."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _db_path(repo: Path) -> Path:
    return repo / "runs" / "_meta" / "history.sqlite"


def _cmd_last(con: sqlite3.Connection, n: int) -> None:
    rows = con.execute(
        "SELECT id, finished_at, best_score, best_pixel_error, preprocess_mode, score_version "
        "FROM experiments ORDER BY id DESC LIMIT ?",
        (int(n),),
    ).fetchall()
    for r in rows:
        print(
            f"id={r['id']} finished={r['finished_at']} score={r['best_score']:.6f} "
            f"px={r['best_pixel_error']:.6f} pre={r['preprocess_mode']} ver={r['score_version']}"
        )


def _cmd_compare(con: sqlite3.Connection, a: int, b: int) -> int:
    ra = con.execute("SELECT * FROM experiments WHERE id=?", (a,)).fetchone()
    rb = con.execute("SELECT * FROM experiments WHERE id=?", (b,)).fetchone()
    if not ra or not rb:
        print("[META] id no encontrado")
        return 2
    print(f"A id={a} score={ra['best_score']} px={ra['best_pixel_error']}")
    print(f"B id={b} score={rb['best_score']} px={rb['best_pixel_error']}")
    print(f"delta_score={float(rb['best_score']) - float(ra['best_score']):.6f}")
    print(
        f"delta_pixel_error={float(rb['best_pixel_error']) - float(ra['best_pixel_error']):.6f}"
    )
    return 0


def _cmd_regressions(con: sqlite3.Connection, pct: float) -> None:
    """Lista corridas con score > (1+pct/100) * mejor historico por mismo input_hash."""
    rows = con.execute(
        "SELECT id, input_hash, best_score, finished_at FROM experiments ORDER BY id ASC"
    ).fetchall()
    best: dict[str, float] = {}
    for r in rows:
        ih = r["input_hash"] or ""
        sc = float(r["best_score"])
        if not ih:
            continue
        if ih not in best:
            best[ih] = sc
        else:
            b = best[ih]
            thr = b * (1.0 + pct / 100.0)
            if sc > thr:
                print(
                    f"[REG] id={r['id']} finished={r['finished_at']} score={sc:.6f} "
                    f"baseline_min={b:.6f} input_hash={ih[:12]}..."
                )
            best[ih] = min(b, sc)


def _cmd_baseline(con: sqlite3.Connection, input_hash: str) -> None:
    row = con.execute(
        "SELECT id, best_score, best_pixel_error, best_params_json, finished_at "
        "FROM experiments WHERE input_hash = ? ORDER BY best_score ASC LIMIT 1",
        (input_hash,),
    ).fetchone()
    if not row:
        print("[META] sin filas para ese input_hash")
        return
    print(
        f"[META] baseline id={row['id']} score={row['best_score']:.6f} "
        f"px={row['best_pixel_error']:.6f} finished={row['finished_at']}"
    )
    print(f"  params={row['best_params_json']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis de historial meta")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument(
        "--last",
        type=int,
        default=10,
        metavar="N",
        help="Listar ultimos N experimentos (0 = omitir)",
    )
    ap.add_argument("--compare", type=int, nargs=2, metavar=("ID_A", "ID_B"), help="Comparar dos ids")
    ap.add_argument(
        "--regressions",
        action="store_true",
        help="Corridas >5%% peores que el minimo historico por input_hash",
    )
    ap.add_argument(
        "--regression-pct",
        type=float,
        default=5.0,
        help="Umbral porcentual sobre el minimo por input_hash (default 5)",
    )
    ap.add_argument("--baseline-for", type=str, metavar="INPUT_HASH", help="Mejor experimento por hash de input")
    args = ap.parse_args()
    repo: Path = args.repo
    p = _db_path(repo)
    if not p.is_file():
        print("[META] sin base aun:", p)
        return 0
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    rc = 0
    try:
        if args.last > 0:
            _cmd_last(con, args.last)
        if args.compare is not None:
            rc = _cmd_compare(con, int(args.compare[0]), int(args.compare[1]))
        if args.regressions:
            _cmd_regressions(con, float(args.regression_pct))
        if args.baseline_for:
            _cmd_baseline(con, str(args.baseline_for))
    finally:
        con.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
