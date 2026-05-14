"""Registro best-effort de una corrida de laser_target_match."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from scripts.meta.history import HistoryDB


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def record_experiment(out_dir: Path, args: Namespace, repo: Path) -> None:
    """
    Inserta una fila en history.sqlite leyendo best_report.json.

    Falla en silencio con warning en stderr si algo va mal.
    """
    try:
        br = out_dir / "best_report.json"
        if not br.is_file():
            return
        data: dict[str, Any] = json.loads(br.read_text(encoding="utf-8"))
        inp = Path(str(getattr(args, "input", ""))).expanduser()
        tgt = Path(str(getattr(args, "target", ""))).expanduser()
        if not inp.is_absolute():
            inp = (repo / inp).resolve() if (repo / inp).is_file() else inp
        else:
            inp = inp.resolve()
        if not tgt.is_absolute():
            tgt = (repo / tgt).resolve() if (repo / tgt).is_file() else tgt
        else:
            tgt = tgt.resolve()
        ih = _sha256_file(inp) if inp.is_file() else ""
        th = _sha256_file(tgt) if tgt.is_file() else ""
        db_path = repo / "runs" / "_meta" / "history.sqlite"
        db = HistoryDB(db_path)
        best_params = {
            k: data.get(k)
            for k in (
                "algorithm",
                "invert",
                "threshold",
                "contrast",
                "brightness",
                "gamma",
                "autocontrast",
                "sharpen",
            )
            if k in data
        }
        rows = []
        for i, row in enumerate(data.get("top", [])[:20], start=1):
            rows.append(
                {
                    "param_name": "composite",
                    "param_value": json.dumps(
                        {k: row.get(k) for k in ("algorithm", "threshold", "contrast", "brightness", "gamma")},
                        sort_keys=True,
                    ),
                    "rank_in_run": i,
                    "score": float(row.get("score", 0.0)),
                }
            )
        wall = float(data.get("wallclock_seconds", 0.0) or 0.0)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - wall)) if wall > 0 else ""
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        eid = db.insert_experiment(
            started_at=started or finished,
            finished_at=finished,
            input_path=str(inp),
            input_hash=ih,
            target_path=str(tgt),
            target_hash=th,
            preprocess_mode=str(getattr(args, "preprocess_mode", "")),
            score_version=str(getattr(args, "score_version", "v1")),
            sampling=str(getattr(args, "sampling", "sobol")),
            n_planned=int(getattr(args, "n", 0)),
            n_evaluated=int(data.get("n_evaluated", 0) or 0) or int(getattr(args, "n", 0)),
            best_score=float(data.get("score", 0.0)),
            best_pixel_error=float(data.get("pixel_error", 0.0)),
            best_params_json=json.dumps(best_params, ensure_ascii=False),
            wallclock_seconds=wall,
            cli_args_json=json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False)[:8000],
            git_sha=_git_sha(repo),
            notes="",
        )
        if rows:
            db.insert_param_stats(eid, rows)
        db.close()
    except Exception as exc:
        import sys

        print(f"[META] record_experiment omitido: {exc}", file=sys.stderr)
