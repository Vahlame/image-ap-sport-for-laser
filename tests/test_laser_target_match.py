from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "laser_target_match.py"


def load_target_module():
    spec = importlib.util.spec_from_file_location("laser_target_match", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_refine_candidates_expand_winner_neighborhood(tmp_path: Path) -> None:
    target = load_target_module()
    db_path = tmp_path / "match.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE matches (
                algorithm TEXT NOT NULL,
                invert INTEGER NOT NULL,
                threshold INTEGER NOT NULL,
                contrast REAL NOT NULL,
                brightness REAL NOT NULL,
                gamma REAL NOT NULL,
                autocontrast REAL NOT NULL,
                sharpen REAL NOT NULL,
                score REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO matches (
                algorithm, invert, threshold, contrast, brightness,
                gamma, autocontrast, sharpen, score
            )
            VALUES ('sierra3_serpentine', 1, 76, 0.55, 28.0, 1.05, 2.0, 0.0, 0.1802)
            """
        )

    candidates = target.local_refine_candidates(db_path, top_k=1, limit=80)
    algorithms = {candidate.algorithm for candidate in candidates}
    thresholds = {candidate.threshold for candidate in candidates}

    assert len(candidates) == 80
    assert "sierra3_serpentine" in algorithms
    assert "two_pass_blue_then_sierra3" in algorithms
    assert min(thresholds) < 76
    assert max(thresholds) > 76


def test_sort_candidates_threshold_proximity_orders_by_distance() -> None:
    target = load_target_module()
    gray = np.full((32, 32), 128.0, dtype=np.float64)
    white_ratio = 0.5
    cand_a = target.Candidate("threshold", False, 50, 1.0, 0.0, 1.0, 1.0, 0.0)
    cand_b = target.Candidate("threshold", False, 126, 1.0, 0.0, 1.0, 1.0, 0.0)
    cand_c = target.Candidate("threshold", False, 127, 1.0, 0.0, 1.0, 1.0, 0.0)
    otsu = int(target.otsu_threshold(gray))
    q_thr = int(np.clip(np.quantile(gray, 1.0 - float(white_ratio)), 1, 254))
    mid = int(round((otsu + q_thr) / 2))

    def expected_key(c: target.Candidate) -> tuple[int, int, int, str]:
        t = int(c.threshold)
        return (abs(t - mid), abs(t - otsu), abs(t - q_thr), c.algorithm)

    inp = [cand_a, cand_b, cand_c]
    out = target.sort_candidates_threshold_proximity(inp, gray, white_ratio)
    assert out == sorted(inp, key=expected_key)
