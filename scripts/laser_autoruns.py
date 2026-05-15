#!/usr/bin/env python3
"""
Ejecuta varias corridas de laser_target_match.py y escribe un resumen JSON.

Salida por sesión (runs/_autoruns/session_YYYYMMDD_HHMMSS/): RUN_MANIFEST.json (v2),
summary.json, README_SESSION.md, una carpeta por corrida. En runs/_autoruns/latest_session.json
apunta a la última sesión terminada.

Uso:
  python scripts/laser_autoruns.py
  python scripts/laser_autoruns.py --workers 0
  python scripts/laser_autoruns.py --quick --workers 8
  python scripts/laser_autoruns.py --v4-only --refine-from-best --refine-from-best-n 320 \\
      --refine-from-best-top 36 --n-base 400 --max-side 384 --workers 4 --explore-brutal \\
      --out-root runs/_autoruns_v4
  python scripts/laser_autoruns.py --n-base 720 --max-side 384 --top-report 28 --from-db-top 36 --workers 6
  LASER_LPIPS_DEVICE=cuda python scripts/laser_autoruns.py   # v4 en GPU (Windows: $env:LASER_LPIPS_DEVICE='cuda')
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import laser_runtime_env

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
_MATCH = _SCRIPT_DIR / "laser_target_match.py"

_SCORE_LINE = re.compile(r"Mejor:\s+.*score=([\d.]+)", re.MULTILINE)


@dataclass
class RunResult:
    name: str
    score_version: str
    out_dir: str
    exit_code: int
    seconds: float
    best_score: float | None
    best_from_sqlite: float | None
    stderr_tail: str


def _parse_stdout_best(stdout: str) -> float | None:
    m = _SCORE_LINE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


def _git_short_rev(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _torch_cuda_snapshot() -> dict[str, object]:
    snap: dict[str, object] = {}
    try:
        import torch

        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001 — manifest debe sobrevivir a imports rotos
        snap["torch_import_error"] = str(exc)
    return snap


def _lpips_mode_safe() -> str:
    try:
        return laser_runtime_env.lpips_device_mode()
    except ValueError as exc:
        return f"invalid:{exc}"


def _write_run_manifest(
    session: Path,
    *,
    args: argparse.Namespace,
    n: int,
    ms: int,
    refine_db_used: bool,
    planned_runs: list[dict[str, str]],
) -> None:
    manifest: dict[str, object] = {
        "schema": "laser_autoruns_run_manifest/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(_REPO),
        "git_rev_short": _git_short_rev(_REPO),
        "argv": sys.argv,
        "python": {
            "executable": str(Path(args.python).resolve()),
            "version": sys.version.split()[0],
            "full": sys.version.replace("\n", " "),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "env_LASER_LPIPS_DEVICE": os.environ.get("LASER_LPIPS_DEVICE"),
        "laser_lpips_device_normalized": _lpips_mode_safe(),
        "inputs": {
            "input_image": str(args.input.resolve()),
            "target_image": str(args.target.resolve()),
            "refine_db": str(args.refine_db.resolve()) if refine_db_used else None,
        },
        "run_parameters": {
            "n_candidates": n,
            "max_side_default": ms,
            "workers": args.workers,
            "quick": args.quick,
            "top_report": args.top_report,
            "from_db_top": args.from_db_top,
            "out_root": str(args.out_root.resolve()),
            "v4_only": bool(args.v4_only),
            "explore_brutal_extra_run": bool(args.explore_brutal),
            "refine_from_best": bool(args.refine_from_best),
            "refine_from_best_n": args.refine_from_best_n,
            "refine_from_best_top": args.refine_from_best_top,
        },
        "planned_runs": planned_runs,
        "torch_cuda": _torch_cuda_snapshot(),
    }
    (session / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _finalize_run_manifest(
    session: Path,
    *,
    wall_seconds_total: float,
    exit_code: int,
    failed_runs: list[str],
) -> None:
    path = session / "RUN_MANIFEST.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    data["completed_utc"] = datetime.now(timezone.utc).isoformat()
    data["wall_seconds_total"] = round(wall_seconds_total, 3)
    data["exit_code"] = exit_code
    data["failed_runs"] = failed_runs
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_latest_session_pointer(out_root: Path, session: Path, *, exit_code: int, all_passed: bool) -> None:
    payload = {
        "schema": "laser_autoruns_latest_session/v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "exit_code": exit_code,
        "all_passed": all_passed,
        "summary_json": str(session / "summary.json"),
        "manifest_json": str(session / "RUN_MANIFEST.json"),
        "readme_md": str(session / "README_SESSION.md"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sqlite_min_score(db: Path) -> float | None:
    if not db.is_file():
        return None
    try:
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT MIN(score) FROM matches").fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
    except sqlite3.Error:
        return None


def _common_replace_n(common: list[str], new_n: int) -> list[str]:
    out = list(common)
    try:
        i = out.index("--n")
        out[i + 1] = str(int(new_n))
    except (ValueError, IndexError):
        pass
    return out


def _strip_local_refine_args(extra: list[str]) -> list[str]:
    """Quita --from-db / --from-db-top / --n para rearmar un refine desde el sqlite del mejor run."""
    skip_next = frozenset({"--from-db", "--from-db-top", "--n"})
    out: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in skip_next:
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _run_one(
    python: Path,
    out_dir: Path,
    common: list[str],
    extra: list[str],
    name: str,
    score_version: str,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), str(_MATCH), *common, "--score-version", score_version, *extra, "--out", str(out_dir)]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_REPO), env=laser_runtime_env.child_process_env()
    )
    elapsed = time.perf_counter() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    best = _parse_stdout_best(stdout)
    sqlite_best = _sqlite_min_score(out_dir / "match.sqlite")
    tail = (stderr + "\n" + stdout)[-4000:]
    return RunResult(
        name=name,
        score_version=score_version,
        out_dir=str(out_dir),
        exit_code=int(proc.returncode),
        seconds=round(elapsed, 3),
        best_score=best,
        best_from_sqlite=sqlite_best,
        stderr_tail=tail,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Varias corridas autonomas de laser_target_match + resumen JSON")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--input", type=Path, default=_REPO / "runs/references/foto_objetivo_sin_procesar.jpeg")
    ap.add_argument("--target", type=Path, default=_REPO / "runs/references/target_imagr_acrylic.png")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = delegar a laser_target_match (v4+CUDA auto-limita GPU; v2/v3 usan todos los núcleos). "
        ">0 fuerza --workers en cada subcorrida.",
    )
    ap.add_argument("--n-base", type=int, default=160, help="Candidatos por corrida (--quick divide entre 2)")
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--top-report", type=int, default=12, help="Mejores filas PNG por corrida (pasa a laser_target_match)")
    ap.add_argument(
        "--from-db-top",
        type=int,
        default=20,
        help="--from-db-top para la corrida 07_fromdb (si existe refine-db)",
    )
    ap.add_argument("--quick", action="store_true", help="Mitad de --n-base y omite corridas mas pesadas")
    ap.add_argument(
        "--refine-db",
        type=Path,
        default=_REPO / "runs/_campaign_push/r3_pinch_thr80/match.sqlite",
        help="Si existe, se anade corrida from-db; si no, se omite.",
    )
    ap.add_argument("--out-root", type=Path, default=_REPO / "runs/_autoruns")
    ap.add_argument(
        "--v4-only",
        action="store_true",
        help="Solo corridas con --score-version v4 (grid/sobol × default/acrylic, BT.709, opcional hi-side y from-db en v4).",
    )
    ap.add_argument(
        "--explore-brutal",
        action="store_true",
        help="Añade una corrida v4 extra con --explore-brutal (perturbaciones tras plateau más amplias).",
    )
    ap.add_argument(
        "--refine-from-best",
        action="store_true",
        help="Solo con --v4-only: al terminar las corridas, ejecuta 99_v4_refine_from_session_best "
        "desde match.sqlite del mejor run de la sesión (Sobol + mismos flags que el ganador).",
    )
    ap.add_argument(
        "--refine-from-best-n",
        type=int,
        default=320,
        help="Valor de --n para refine-from-best (default 320).",
    )
    ap.add_argument(
        "--refine-from-best-top",
        type=int,
        default=36,
        help="--from-db-top para refine-from-best (default 36).",
    )
    args = ap.parse_args()
    laser_runtime_env.apply_lpips_default_process_env()
    laser_runtime_env.coerce_lpips_env_if_cuda_unavailable()

    if args.refine_from_best and not args.v4_only:
        print("[WARN] --refine-from-best solo tiene efecto con --v4-only; se ignora.", flush=True)

    if not _MATCH.is_file():
        print(f"No se encuentra {_MATCH}", file=sys.stderr)
        return 2
    if not args.input.is_file() or not args.target.is_file():
        print("Faltan imagenes en runs/references (usa --input/--target).", file=sys.stderr)
        return 2

    n = max(24, args.n_base // 2) if args.quick else max(40, args.n_base)
    ms = min(args.max_side, 280) if args.quick else args.max_side

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session = args.out_root / f"session_{stamp}"
    session.mkdir(parents=True, exist_ok=True)

    refine_db_used = args.refine_db.is_file()

    common = [
        "--input",
        str(args.input),
        "--target",
        str(args.target),
        "--n",
        str(n),
        "--max-side",
        str(ms),
        "--register",
        "none",
        "--guided-explore",
        "--sort-candidates",
        "threshold-proximity",
        "--top-report",
        str(args.top_report),
        "--preprocess-mode",
        "none",
    ]
    if args.workers > 0:
        common.extend(["--workers", str(args.workers)])

    specs: list[tuple[str, list[str], str]]
    if args.v4_only:
        specs = [
            ("01_v4_grid_default", ["--sampling", "grid", "--search-preset", "default"], "v4"),
            ("02_v4_grid_acrylic", ["--sampling", "grid", "--search-preset", "acrylic"], "v4"),
            ("03_v4_sobol_default", ["--sampling", "sobol", "--search-preset", "default"], "v4"),
            ("04_v4_sobol_acrylic", ["--sampling", "sobol", "--search-preset", "acrylic"], "v4"),
            ("05_v4_grid_bt709", ["--sampling", "grid", "--search-preset", "default", "--luma", "bt709"], "v4"),
        ]
        if not args.quick:
            hi = max(ms + 48, 384)
            specs.append(
                (
                    "06_v4_sobol_acrylic_hi_side",
                    ["--sampling", "sobol", "--search-preset", "acrylic", "--max-side", str(min(hi, 520))],
                    "v4",
                )
            )
        if refine_db_used:
            specs.append(
                (
                    "07_v4_fromdb_refine",
                    [
                        "--from-db",
                        str(args.refine_db),
                        "--from-db-top",
                        str(args.from_db_top),
                        "--n",
                        str(max(48, n // 2)),
                        "--sampling",
                        "grid",
                        "--search-preset",
                        "acrylic",
                    ],
                    "v4",
                )
            )
        else:
            print(f"[SKIP] refine-db inexistente: {args.refine_db}", flush=True)

        if not args.quick:
            specs.append(
                (
                    "08_v4_sauvola_grid_acrylic",
                    ["--sampling", "grid", "--search-preset", "acrylic", "--preprocess-mode", "sauvola"],
                    "v4",
                )
            )

        if args.explore_brutal:
            specs.append(
                (
                    "09_v4_sobol_acrylic_explore_brutal",
                    ["--explore-brutal", "--sampling", "sobol", "--search-preset", "acrylic"],
                    "v4",
                )
            )
    else:
        specs = [
            ("01_grid_default", ["--sampling", "grid", "--search-preset", "default"], "v2"),
            ("02_grid_acrylic", ["--sampling", "grid", "--search-preset", "acrylic"], "v2"),
            ("03_sobol_default", ["--sampling", "sobol", "--search-preset", "default"], "v2"),
            ("04_sobol_acrylic", ["--sampling", "sobol", "--search-preset", "acrylic"], "v2"),
            ("05_grid_bt709", ["--sampling", "grid", "--search-preset", "default", "--luma", "bt709"], "v2"),
            ("09_grid_v3", ["--sampling", "grid", "--search-preset", "default"], "v3"),
            ("10_grid_v4", ["--sampling", "grid", "--search-preset", "default"], "v4"),
        ]
        if not args.quick:
            specs.append(("06_sobol_maxside340", ["--sampling", "sobol", "--search-preset", "default", "--max-side", "340"], "v2"))

        if refine_db_used:
            specs.append(
                (
                    "07_fromdb_r3_top20",
                    [
                        "--from-db",
                        str(args.refine_db),
                        "--from-db-top",
                        str(args.from_db_top),
                        "--n",
                        str(max(24, n // 2)),
                        "--sampling",
                        "grid",
                    ],
                    "v2",
                )
            )
        else:
            print(f"[SKIP] refine-db inexistente: {args.refine_db}", flush=True)

        if not args.quick:
            specs.append(("08_sauvola_grid", ["--sampling", "grid", "--search-preset", "default", "--preprocess-mode", "sauvola"], "v2"))

    spec_extras = {name: list(extra) for name, extra, _sv in specs}

    planned_runs = [{"name": name, "score_version": sv} for name, _, sv in specs]
    if args.v4_only and args.refine_from_best:
        planned_runs.append({"name": "99_v4_refine_from_session_best", "score_version": "v4"})
    _write_run_manifest(
        session,
        args=args,
        n=n,
        ms=ms,
        refine_db_used=refine_db_used,
        planned_runs=planned_runs,
    )

    results: list[RunResult] = []
    wmsg = str(args.workers) if args.workers > 0 else "delegado (v4+CUDA cap auto)"
    print(f"[AUTORUN] session={session} n={n} max-side={ms} workers={wmsg}", flush=True)
    if args.v4_only:
        print("[AUTORUN] modo=v4-only (solo métrica perceptual LPIPS/blur)", flush=True)
    if args.v4_only and args.refine_from_best:
        print(
            f"[AUTORUN] refine-from-best al final (n={args.refine_from_best_n}, "
            f"from-db-top={args.refine_from_best_top})",
            flush=True,
        )
    t_wall0 = time.perf_counter()
    for name, extra, sv in specs:
        print(f"[AUTORUN] -> {name} score-version={sv}", flush=True)
        r = _run_one(args.python, session / name, common, extra, name, sv)
        results.append(r)
        sb = r.best_from_sqlite if r.best_from_sqlite is not None else r.best_score
        print(f"         exit={r.exit_code} best_sqlite={sb} t={r.seconds}s", flush=True)
        if r.exit_code != 0 and r.stderr_tail.strip():
            print(f"         --- stderr+stdout tail ---\n{r.stderr_tail}\n         --- end ---", flush=True)

    loop_elapsed = time.perf_counter() - t_wall0
    wall_seconds_total = loop_elapsed

    refine_from_best_meta: dict[str, object] | None = None
    if args.v4_only and args.refine_from_best:
        successes = [
            r
            for r in results
            if r.exit_code == 0
            and (Path(r.out_dir).resolve() / "match.sqlite").is_file()
            and (r.best_from_sqlite is not None or r.best_score is not None)
        ]
        if not successes:
            print("[SKIP] refine-from-best: no hay corridas exitosas con match.sqlite y score.", flush=True)
        else:

            def _score_key(rr: RunResult) -> float:
                sc = rr.best_from_sqlite if rr.best_from_sqlite is not None else rr.best_score
                return float(sc) if sc is not None else float("inf")

            best_r = min(successes, key=_score_key)
            base_extra = _strip_local_refine_args(spec_extras.get(best_r.name, []))
            db_path = Path(best_r.out_dir).resolve() / "match.sqlite"
            refine_extra = [
                *base_extra,
                "--from-db",
                str(db_path),
                "--from-db-top",
                str(args.refine_from_best_top),
                "--sampling",
                "sobol",
            ]
            if args.explore_brutal and "--explore-brutal" not in refine_extra:
                refine_extra.insert(0, "--explore-brutal")
            refine_common = _common_replace_n(common, max(48, args.refine_from_best_n))
            name99 = "99_v4_refine_from_session_best"
            print(f"[AUTORUN] -> {name99} score-version=v4 (desde mejor={best_r.name})", flush=True)
            t_ref = time.perf_counter()
            r99 = _run_one(args.python, session / name99, refine_common, refine_extra, name99, "v4")
            results.append(r99)
            wall_seconds_total += time.perf_counter() - t_ref
            sb99 = r99.best_from_sqlite if r99.best_from_sqlite is not None else r99.best_score
            print(f"         exit={r99.exit_code} best_sqlite={sb99} t={r99.seconds}s", flush=True)
            if r99.exit_code != 0 and r99.stderr_tail.strip():
                print(f"         --- stderr+stdout tail ---\n{r99.stderr_tail}\n         --- end ---", flush=True)

            refine_from_best_meta = {
                "source_run": best_r.name,
                "from_sqlite": str(db_path),
                "refine_out_dir": str((session / name99).resolve()),
                "refine_n": int(args.refine_from_best_n),
                "refine_from_db_top": int(args.refine_from_best_top),
            }
    all_passed = not failed_runs

    by_ver: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for r in results:
        sc = r.best_from_sqlite if r.best_from_sqlite is not None else r.best_score
        if sc is not None:
            by_ver[r.score_version].append((r.name, sc, r.out_dir))

    winners_by_score_version: dict[str, dict[str, object]] = {}
    for ver, lst in sorted(by_ver.items()):
        lst.sort(key=lambda t: t[1])
        winners_by_score_version[ver] = {"name": lst[0][0], "score": lst[0][1], "dir": lst[0][2]}

    summary = {
        "session": str(session),
        "manifest": str(session / "RUN_MANIFEST.json"),
        "n": n,
        "max_side_default": ms,
        "top_report": args.top_report,
        "from_db_top": args.from_db_top,
        "quick": bool(args.quick),
        "v4_only": bool(args.v4_only),
        "run_count": len(results),
        "wall_seconds_total": round(wall_seconds_total, 3),
        "failed_runs": failed_runs,
        "all_passed": all_passed,
        "runs": [asdict(r) for r in results],
        "winners_by_score_version": winners_by_score_version,
    }
    if args.v4_only:
        ranked: list[dict[str, object]] = []
        for r in results:
            sc = r.best_from_sqlite if r.best_from_sqlite is not None else r.best_score
            if sc is not None:
                ranked.append({"name": r.name, "score": sc, "dir": r.out_dir, "exit_code": r.exit_code})
        if ranked:
            ranked.sort(key=lambda row: float(row["score"]))  # type: ignore[arg-type]
            summary["best_v4_across_runs"] = ranked[0]
            summary["v4_ranked"] = ranked
    if refine_from_best_meta is not None:
        summary["refine_from_best"] = refine_from_best_meta
    (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    exit_code = 0 if all_passed else 1
    _finalize_run_manifest(session, wall_seconds_total=wall_seconds_total, exit_code=exit_code, failed_runs=failed_runs)
    _write_latest_session_pointer(args.out_root.resolve(), session, exit_code=exit_code, all_passed=all_passed)

    manifest_path = session / "RUN_MANIFEST.json"
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        utc_start = str(manifest_data.get("created_utc", ""))
    except OSError:
        utc_start = ""
    readme = session / "README_SESSION.md"
    winners_md = "\n".join(
        f"- **{ver}**: `{w['name']}` -> `{float(w['score']):.6f}` (`{w['dir']}`)"
        for ver, w in sorted(winners_by_score_version.items())
    )
    table_lines = [
        "| Run | Score ver | Exit | Best | s |",
        "|-----|-----------|------|------|---|",
    ]
    for r in results:
        sc = r.best_from_sqlite if r.best_from_sqlite is not None else r.best_score
        sc_s = f"{sc:.6f}" if sc is not None else ""
        table_lines.append(f"| `{r.name}` | {r.score_version} | {r.exit_code} | {sc_s} | {r.seconds} |")

    v4_sess_block: list[str] = []
    if args.v4_only:
        bx = summary.get("best_v4_across_runs")
        if isinstance(bx, dict):
            v4_sess_block = [
                "## Mejor v4 (global en esta sesión)",
                "",
                f"- `{bx['name']}` -> **{float(bx['score']):.6f}** (`{bx['dir']}`)",
                "",
            ]
            rf = summary.get("refine_from_best")
            if isinstance(rf, dict):
                v4_sess_block.extend(
                    [
                        "**Refine:** corrida `99_v4_refine_from_session_best` desde el mejor run "
                        f"`{rf.get('source_run', '')}` (`from-db`: `{rf.get('from_sqlite', '')}`).",
                        "",
                    ]
                )

    readme_done = datetime.now(timezone.utc).isoformat()
    readme.write_text(
        "\n".join(
            [
                "# Autorun session",
                "",
                f"- **Inicio (UTC, manifiesto)**: `{utc_start}`",
                f"- **README generado (UTC)**: `{readme_done}`",
                f"- **Directorio**: `{session}`",
                f"- **Wall clock (todas las corridas)**: `{round(wall_seconds_total, 3)}` s",
                f"- **Estado**: {'OK (todas exit 0)' if all_passed else 'CON FALLOS — ver `failed_runs` en summary.json'}",
                "- **Manifiesto**: `RUN_MANIFEST.json` (schema v2: plan + cierre con tiempos / fallos)",
                "- **Resumen máquina**: `summary.json`",
                "- **Última sesión (repo)**: `../latest_session.json` relativo a esta carpeta padre `runs/_autoruns/`",
                "",
                "## Corridas",
                "",
                *table_lines,
                "",
                *v4_sess_block,
                "## Mejores por versión de score",
                "",
                winners_md if winners_md else "_Sin ganadores parseados._",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[AUTORUN] summary -> {session / 'summary.json'}", flush=True)
    print(f"[AUTORUN] latest pointer -> {args.out_root.resolve() / 'latest_session.json'}", flush=True)
    for ver, w in winners_by_score_version.items():
        print(f"[AUTORUN] mejor ({ver}): {w['name']} score={float(w['score']):.6f}", flush=True)
    if args.v4_only and "best_v4_across_runs" in summary:
        b = summary["best_v4_across_runs"]
        print(
            f"[AUTORUN] mejor v4 global en sesión: {b['name']} score={float(b['score']):.6f}",
            flush=True,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
