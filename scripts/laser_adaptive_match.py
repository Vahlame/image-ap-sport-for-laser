#!/usr/bin/env python3
"""
Encadena varias corridas de laser_target_match.py: tras cada ronda lee el mejor score,
detecta si la corrida termino antes de agotar --n (pocos candidatos unicos / cola corta)
y ajusta parametros de la siguiente (refine-db, anchura de refine, brutal explore,
pinch en capa densa, semilla Sobol, preset acrylic, etc.).

Uso tipico:
  python scripts/laser_adaptive_match.py \\
    --input runs/references/foto_objetivo_sin_procesar.jpeg \\
    --target runs/references/target_imagr_acrylic.png \\
    --adapt-root runs/_adaptive_demo \\
    --rounds 5 \\
    --base-n 450 \\
    --max-side 340

Ronda 0: exploracion amplia (sin --refine-db). Rondas >= 1: --refine-db apunta al mejor
SQLite global visto hasta el momento; si no hubo mejora, se endurece la estrategia
(brutal, refine-breadth, refine-top, pinch mas ancho, cambio sampling/preset).

Mejoras marginales (delta muy pequeno en score): acumulan micro-estancamiento y suben el
escalamiento igual que un plateau real (--marginal-gain-abs / --marginal-gain-rel).

Periodicamente (--explore-refresh-every) puede forzar una ronda solo de exploracion
(con pinch de umbral) para salir de un attractor local del refine.

Opcional: pasar argumentos extra al matcher despues de -- :
  python scripts/laser_adaptive_match.py ... -- --score-version v3 --luma bt709

PowerShell (no uses ^; usa backtick ` o una sola linea). No pegues el texto literal "..." — son rutas y flags reales:
  python scripts/laser_adaptive_match.py `
    --input runs/references/foto_objetivo_sin_procesar.jpeg `
    --target runs/references/target_imagr_acrylic.png `
    --adapt-root runs/_mi_adaptive `
    --rounds 6 --base-n 400 --max-side 340 `
    --explore-refresh-every 4 --explore-refresh-min-escalation 2
"""

from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import laser_runtime_env

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
_MATCH = _SCRIPT_DIR / "laser_target_match.py"


@dataclass
class RoundStats:
    round_index: int
    out_dir: str
    exit_code: int
    seconds: float
    requested_n: int
    eval_count: int | None
    best_score: float | None
    best_threshold: int | None
    early_finish: bool
    notes: str = ""


@dataclass
class AdaptiveKnobs:
    n: int
    max_side: int
    sampling: str
    search_preset: str
    score_version: str
    explore_brutal: bool
    sort_candidates: str
    sobol_seed: int
    dense_min: int | None
    dense_max: int | None
    refine_top: int
    refine_breadth: str
    refine_best_per_algorithm: bool
    workers: int
    register: str
    luma: str
    preprocess_mode: str
    restart_candidates: int | None
    worker_recycle_tasks: int
    extra_args: list[str] = field(default_factory=list)


def _sqlite_stats(db: Path) -> tuple[int | None, float | None]:
    if not db.is_file():
        return None, None
    try:
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT COUNT(*), MIN(score) FROM matches").fetchone()
        if row is None:
            return None, None
        cnt = int(row[0]) if row[0] is not None else None
        best = float(row[1]) if row[1] is not None else None
        return cnt, best
    except sqlite3.Error:
        return None, None


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


def _build_cmd(
    python: Path,
    inp: Path,
    tgt: Path,
    out_dir: Path,
    knobs: AdaptiveKnobs,
    refine_db: Path | None,
) -> list[str]:
    cmd: list[str] = [
        str(python),
        str(_MATCH),
        "--input",
        str(inp),
        "--target",
        str(tgt),
        "--out",
        str(out_dir),
        "--n",
        str(int(knobs.n)),
        "--max-side",
        str(int(knobs.max_side)),
        "--workers",
        str(int(knobs.workers)),
        "--worker-recycle-tasks",
        str(int(knobs.worker_recycle_tasks)),
        "--guided-explore",
        "--sort-candidates",
        knobs.sort_candidates,
        "--sampling",
        knobs.sampling,
        "--search-preset",
        knobs.search_preset,
        "--score-version",
        knobs.score_version,
        "--register",
        knobs.register,
        "--luma",
        knobs.luma,
        "--preprocess-mode",
        knobs.preprocess_mode,
        "--sobol-seed",
        str(int(knobs.sobol_seed)),
        "--top-report",
        "24",
    ]
    if knobs.explore_brutal:
        cmd.append("--explore-brutal")
    else:
        cmd.append("--no-explore-brutal")
    if knobs.restart_candidates is not None:
        cmd += ["--restart-candidates", str(int(knobs.restart_candidates))]

    if refine_db is not None:
        cmd += [
            "--refine-db",
            str(refine_db),
            "--refine-top",
            str(int(knobs.refine_top)),
            "--refine-breadth",
            knobs.refine_breadth,
        ]
        if knobs.refine_best_per_algorithm:
            cmd.append("--refine-best-per-algorithm")
    else:
        if knobs.dense_min is not None and knobs.dense_min > 0:
            cmd += ["--dense-threshold-min", str(int(knobs.dense_min))]
        if knobs.dense_max is not None and knobs.dense_max > 0:
            cmd += ["--dense-threshold-max", str(int(knobs.dense_max))]

    cmd.extend(knobs.extra_args)
    return cmd


def _adapt_knobs(
    knobs: AdaptiveKnobs,
    *,
    escalation: int,
    global_thr: int | None,
    early_finish: bool,
    eval_ratio: float,
    n_cap: int,
    pinch_base: int,
) -> AdaptiveKnobs:
    """Copia knobs y aplica heuristica para la siguiente ronda.

    escalation combina rondas sin mejora y micro-estancamiento (mejoras minusculas).
    """
    k = AdaptiveKnobs(**asdict(knobs))

    if early_finish or eval_ratio < 0.92:
        k.n = min(n_cap, max(k.n, int(k.n * 1.2) + 48))
        k.sobol_seed = (k.sobol_seed + 37) % (2**31 - 1)
        if k.sampling == "sobol":
            k.sampling = "grid"
        else:
            k.sampling = "sobol"

    if escalation >= 1:
        k.explore_brutal = True
        k.refine_top = min(24, max(k.refine_top, 8 + escalation * 2))
        k.refine_best_per_algorithm = True
        k.restart_candidates = min(6000, 900 + escalation * 400)

    if escalation >= 2:
        k.search_preset = "acrylic"

    if escalation <= 1:
        k.refine_breadth = "normal"
    elif escalation == 2:
        k.refine_breadth = "deep"
    else:
        k.refine_breadth = "max"

    if escalation >= 1 and eval_ratio >= 0.95:
        k.n = min(n_cap, int(k.n * 1.15) + 32)

    if escalation >= 3:
        k.luma = "bt709"

    if escalation >= 6:
        k.preprocess_mode = "niblack"
    elif escalation >= 4:
        k.preprocess_mode = "sauvola"

    if global_thr is not None:
        radius = pinch_base + min(34, escalation * 6)
        k.dense_min = max(1, global_thr - radius)
        k.dense_max = min(254, global_thr + radius)

    return k


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Varias rondas adaptativas de laser_target_match (refine + pinch + brutal segun resultado)."
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--adapt-root", type=Path, required=True, help="Carpeta de sesion (se crean round_001, ...)")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--base-n", type=int, default=400)
    ap.add_argument("--n-cap", type=int, default=120_000, help="Tope superior de --n al escalar")
    ap.add_argument("--max-side", type=int, default=340)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Passthrough a laser_target_match (0 = nucleos segun env LASER_MATCH_MAX_WORKERS o todos)",
    )
    ap.add_argument(
        "--worker-recycle-tasks",
        type=int,
        default=48,
        help="Passthrough a laser_target_match (multiproceso Py>=3.11). 0=desactivar reciclaje de workers.",
    )
    ap.add_argument("--score-version", choices=("v2", "v3", "v4"), default="v2")
    ap.add_argument("--sampling", choices=("grid", "sobol"), default="sobol")
    ap.add_argument("--search-preset", choices=("default", "acrylic"), default="default")
    ap.add_argument("--seed-sqlite", type=Path, default=None, help="Si existe: ronda 0 usa --refine-db sobre este SQLite")
    ap.add_argument("--min-delta-improve", type=float, default=1e-6, help="Mejora minima de score para resetear estancamiento")
    ap.add_argument(
        "--marginal-gain-abs",
        type=float,
        default=4e-5,
        help="Si la mejora es menor que este delta absoluto (score v2/v3/v4), sube micro-estancamiento",
    )
    ap.add_argument(
        "--marginal-gain-rel",
        type=float,
        default=8e-5,
        help="Si la mejora relativa es menor, sube micro-estancamiento (junto con --marginal-gain-abs)",
    )
    ap.add_argument("--early-ratio", type=float, default=0.88, help="Si evals/n < ratio, se considera termino temprano")
    ap.add_argument("--pinch-base-radius", type=int, default=18, help="Radio inicial +/- umbral para dense_threshold_* en exploracion")
    ap.add_argument(
        "--explore-refresh-every",
        type=int,
        default=0,
        help="Cada N rondas, una corrida solo exploracion (sin --refine-db) si escalation>=explore-refresh-min",
    )
    ap.add_argument(
        "--explore-refresh-min-escalation",
        type=int,
        default=2,
        help="Umbral de escalation para activar explore-refresh-every",
    )
    ap.add_argument("--dry-run", action="store_true", help="Solo imprimir comandos")
    ap.add_argument("--continue-on-error", action="store_true", help="Seguir rondas aunque falle una corrida")
    ap.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Argumentos extra para laser_target_match (ej. -- --score-version v3)",
    )
    args = ap.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()

    if not _MATCH.is_file():
        print(f"No existe {_MATCH}", file=sys.stderr)
        return 2

    inp = args.input if args.input.is_absolute() else (_REPO / args.input).resolve()
    tgt = args.target if args.target.is_absolute() else (_REPO / args.target).resolve()
    if not inp.is_file() or not tgt.is_file():
        print("Faltan --input o --target.", file=sys.stderr)
        return 2

    adapt_root = args.adapt_root if args.adapt_root.is_absolute() else (_REPO / args.adapt_root).resolve()
    adapt_root.mkdir(parents=True, exist_ok=True)

    extra: list[str] = []
    if args.remainder:
        extra = list(args.remainder)
        if extra and extra[0] == "--":
            extra = extra[1:]

    knobs = AdaptiveKnobs(
        n=max(40, int(args.base_n)),
        max_side=int(args.max_side),
        sampling=str(args.sampling),
        search_preset=str(args.search_preset),
        score_version=str(args.score_version),
        explore_brutal=False,
        sort_candidates="threshold-proximity",
        sobol_seed=42,
        dense_min=None,
        dense_max=None,
        refine_top=8,
        refine_breadth="normal",
        refine_best_per_algorithm=False,
        workers=int(args.workers),
        register="none",
        luma="bt601",
        preprocess_mode="none",
        restart_candidates=None,
        worker_recycle_tasks=max(0, int(args.worker_recycle_tasks)),
        extra_args=extra,
    )

    seed_sqlite = args.seed_sqlite
    if seed_sqlite is not None:
        seed_sqlite = seed_sqlite if seed_sqlite.is_absolute() else (_REPO / seed_sqlite).resolve()

    global_best: float | None = None
    global_best_sqlite: Path | None = None
    global_thr: int | None = None
    if seed_sqlite is not None and seed_sqlite.is_file():
        global_best_sqlite = seed_sqlite
        _, sc = _sqlite_stats(seed_sqlite)
        global_best = sc
        global_thr = _read_best_threshold(seed_sqlite.parent)
        print(f"[ADAPT] seed sqlite={seed_sqlite} min_score={global_best}", flush=True)
    stagnation = 0
    micro_stagnation = 0
    history: list[RoundStats] = []

    n_rounds = max(1, int(args.rounds))
    n_cap = max(knobs.n, int(args.n_cap))

    for r in range(n_rounds):
        out_dir = adapt_root / f"round_{r + 1:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        use_refine = global_best_sqlite is not None and global_best_sqlite.is_file()
        refresh_n = int(args.explore_refresh_every)
        refresh_min = int(args.explore_refresh_min_escalation)
        esc_now = stagnation + micro_stagnation
        force_explore = refresh_n > 0 and (r + 1) % refresh_n == 0 and esc_now >= refresh_min
        refine_arg = None if force_explore else (global_best_sqlite if use_refine else None)
        if force_explore:
            print(f"[ADAPT] explore-refresh: ronda {r+1} sin refine-db (escalation={esc_now})", flush=True)

        cmd = _build_cmd(args.python, inp, tgt, out_dir, knobs, refine_arg)
        log_path = out_dir / "adaptive_subprocess.log"
        meta_path = out_dir / "adaptive_round_meta.json"

        print(f"\n[ADAPT] === ronda {r + 1}/{n_rounds} ===", flush=True)
        print(f"[ADAPT] out_dir={out_dir}", flush=True)
        print(f"[ADAPT] cmd={' '.join(shlex.quote(str(x)) for x in cmd)}", flush=True)

        if args.dry_run:
            stats = RoundStats(
                round_index=r,
                out_dir=str(out_dir),
                exit_code=0,
                seconds=0.0,
                requested_n=knobs.n,
                eval_count=None,
                best_score=None,
                best_threshold=None,
                early_finish=False,
                notes="dry-run",
            )
            history.append(stats)
            meta_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
            knobs = _adapt_knobs(
                knobs,
                escalation=0,
                global_thr=global_thr,
                early_finish=False,
                eval_ratio=1.0,
                n_cap=n_cap,
                pinch_base=args.pinch_base_radius,
            )
            continue

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            env=laser_runtime_env.child_process_env(),
        )
        elapsed = time.perf_counter() - t0
        log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

        db = out_dir / "match.sqlite"
        eval_count, best_score = _sqlite_stats(db)
        best_thr = _read_best_threshold(out_dir)

        requested_n = knobs.n
        eval_ratio = float(eval_count or 0) / float(max(1, requested_n))
        early_finish = eval_count is not None and eval_ratio < float(args.early_ratio)
        over_budget = eval_count is not None and requested_n > 0 and eval_count > requested_n

        notes = ""
        if early_finish:
            notes = f"early_finish eval_ratio={eval_ratio:.3f}"
        if over_budget:
            notes = (notes + "; " if notes else "") + f"evals_over_n={eval_count}-{requested_n}"
        plateau_hits = (proc.stderr or "").count("[PLATEAU]")
        if plateau_hits >= 3:
            notes = (notes + "; " if notes else "") + f"plateau_lines={plateau_hits}"

        stats = RoundStats(
            round_index=r,
            out_dir=str(out_dir),
            exit_code=int(proc.returncode),
            seconds=round(elapsed, 3),
            requested_n=requested_n,
            eval_count=eval_count,
            best_score=best_score,
            best_threshold=best_thr,
            early_finish=early_finish,
            notes=notes.strip("; "),
        )
        history.append(stats)
        meta_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")

        print(
            f"[ADAPT] rc={stats.exit_code} t={stats.seconds}s evals={eval_count}/{requested_n} "
            f"best={best_score} early={early_finish}",
            flush=True,
        )

        if proc.returncode != 0:
            print("[ADAPT] la corrida fallo; ver adaptive_subprocess.log", file=sys.stderr)
            if not args.continue_on_error:
                break
            stagnation += 1
            micro_stagnation += 1
            knobs = _adapt_knobs(
                knobs,
                escalation=stagnation + micro_stagnation,
                global_thr=global_thr,
                early_finish=early_finish,
                eval_ratio=eval_ratio,
                n_cap=n_cap,
                pinch_base=args.pinch_base_radius,
            )
            continue

        if best_score is None:
            print("[ADAPT] sin filas en SQLite; abortando cadena.", file=sys.stderr)
            break

        pre_best = global_best
        marginal_abs = float(args.marginal_gain_abs)
        marginal_rel = float(args.marginal_gain_rel)
        improved = global_best is None or best_score < global_best - float(args.min_delta_improve)
        if improved:
            global_best = best_score
            global_best_sqlite = db
            global_thr = best_thr if best_thr is not None else global_thr
            stagnation = 0
            gain = float(pre_best - best_score) if pre_best is not None else marginal_abs * 100.0
            rel_gain = gain / max(abs(pre_best), 1e-9) if pre_best is not None else 1.0
            tiny = pre_best is not None and gain < marginal_abs and rel_gain < marginal_rel
            if tiny:
                micro_stagnation += 1
            else:
                micro_stagnation = 0
            print(
                f"[ADAPT] nuevo mejor global score={global_best:.8f} sqlite={global_best_sqlite} "
                f"micro_stagnation={micro_stagnation} tiny_gain={tiny}",
                flush=True,
            )
        else:
            stagnation += 1
            micro_stagnation += 1
            print(
                f"[ADAPT] sin mejora stagnation={stagnation} micro={micro_stagnation} "
                f"escalation={stagnation + micro_stagnation}",
                flush=True,
            )

        if r + 1 < n_rounds:
            knobs = _adapt_knobs(
                knobs,
                escalation=stagnation + micro_stagnation,
                global_thr=global_thr,
                early_finish=early_finish,
                eval_ratio=eval_ratio,
                n_cap=n_cap,
                pinch_base=args.pinch_base_radius,
            )

    summary_path = adapt_root / "adaptive_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "adapt_root": str(adapt_root),
                "global_best": global_best,
                "global_best_sqlite": str(global_best_sqlite) if global_best_sqlite else None,
                "global_threshold": global_thr,
                "final_stagnation": stagnation,
                "final_micro_stagnation": micro_stagnation,
                "rounds": [asdict(h) for h in history],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[ADAPT] resumen -> {summary_path}", flush=True)
    if global_best is not None:
        print(f"[ADAPT] mejor score global={global_best:.8f}", flush=True)

    if any(h.exit_code != 0 for h in history) and not args.continue_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
