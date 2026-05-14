"""SQLite persistente de experimentos (runs/_meta/history.sqlite por defecto)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    input_path TEXT,
    input_hash TEXT,
    target_path TEXT,
    target_hash TEXT,
    preprocess_mode TEXT,
    score_version TEXT,
    sampling TEXT,
    n_planned INTEGER,
    n_evaluated INTEGER,
    best_score REAL,
    best_pixel_error REAL,
    best_params_json TEXT,
    wallclock_seconds REAL,
    cli_args_json TEXT,
    git_sha TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS param_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    param_name TEXT,
    param_value TEXT,
    rank_in_run INTEGER,
    score REAL
);
CREATE TABLE IF NOT EXISTS regressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT,
    experiment_id INTEGER,
    baseline_experiment_id INTEGER,
    score_delta REAL,
    pixel_error_delta REAL
);
CREATE INDEX IF NOT EXISTS idx_experiments_input ON experiments(input_hash);
CREATE INDEX IF NOT EXISTS idx_experiments_target ON experiments(target_hash);
"""


class HistoryDB:
    """Conexion administrada a la base de historial."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def insert_experiment(
        self,
        *,
        started_at: str,
        finished_at: str,
        input_path: str,
        input_hash: str,
        target_path: str,
        target_hash: str,
        preprocess_mode: str,
        score_version: str,
        sampling: str,
        n_planned: int,
        n_evaluated: int,
        best_score: float,
        best_pixel_error: float,
        best_params_json: str,
        wallclock_seconds: float,
        cli_args_json: str,
        git_sha: str,
        notes: str = "",
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO experiments (
                started_at, finished_at, input_path, input_hash, target_path, target_hash,
                preprocess_mode, score_version, sampling, n_planned, n_evaluated,
                best_score, best_pixel_error, best_params_json, wallclock_seconds,
                cli_args_json, git_sha, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                started_at,
                finished_at,
                input_path,
                input_hash,
                target_path,
                target_hash,
                preprocess_mode,
                score_version,
                sampling,
                n_planned,
                n_evaluated,
                best_score,
                best_pixel_error,
                best_params_json,
                wallclock_seconds,
                cli_args_json,
                git_sha,
                notes,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def insert_param_stats(self, experiment_id: int, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO param_stats (experiment_id, param_name, param_value, rank_in_run, score)
                VALUES (?,?,?,?,?)
                """,
                (
                    experiment_id,
                    row["param_name"],
                    row["param_value"],
                    int(row["rank_in_run"]),
                    float(row["score"]),
                ),
            )
        self._conn.commit()

    def fetch_last(self, n: int) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?",
            (int(n),),
        )
        return list(cur.fetchall())

    def fetch_by_id(self, eid: int) -> Optional[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM experiments WHERE id = ?", (int(eid),))
        row = cur.fetchone()
        return row
