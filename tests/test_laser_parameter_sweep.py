from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "laser_parameter_sweep.py"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("laser_parameter_sweep", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smart_params_reserve_threshold_ladder() -> None:
    sweep = load_sweep_module()
    gray = np.tile(np.arange(256, dtype=np.float64), (16, 1))

    params = sweep.smart_params(gray, np.random.default_rng(7), n=80, threshold_steps=24)

    assert len(params) == 80
    ladder = [p for p in params if p.source == "threshold_ladder"]
    assert len(ladder) >= 24
    assert {p.contrast for p in ladder} == {1.0}
    assert {p.brightness for p in ladder} == {0.0}
    assert len({p.threshold for p in ladder}) >= 24
    assert all(1 <= p.threshold <= 254 for p in params)


def test_cli_writes_sources_to_sqlite_and_manifest(tmp_path: Path) -> None:
    gradient = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
    rgb = np.dstack([gradient, gradient, gradient])
    input_path = tmp_path / "gradient.png"
    Image.fromarray(rgb, mode="RGB").save(input_path)

    out_dir = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--out",
            str(out_dir),
            "--n",
            "40",
            "--seed",
            "9",
            "--threshold-steps",
            "16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Muestra de escalera de umbrales" in result.stdout
    assert "Reporte:" in result.stdout
    assert "Hoja:" in result.stdout

    db_path = out_dir / "sweep.sqlite"
    with sqlite3.connect(db_path) as conn:
        count, distinct_thresholds = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT threshold) FROM runs"
        ).fetchone()
        sources = {
            row[0]
            for row in conn.execute("SELECT DISTINCT source FROM runs ORDER BY source").fetchall()
        }
        neutral_ladder_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM runs
            WHERE source = 'threshold_ladder' AND contrast = 1.0 AND brightness = 0.0
            """
        ).fetchone()[0]

    assert count == 40
    assert distinct_thresholds > 12
    assert "threshold_ladder" in sources
    assert "histogram_anchor_grid" in sources
    assert neutral_ladder_count >= 16

    manifest_rows = [
        json.loads(line)
        for line in (out_dir / "sweep_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest_rows) == 40
    assert {"source", "sequence", "threshold"} <= set(manifest_rows[0])
    assert (out_dir / manifest_rows[0]["output_file"]).is_file()
    assert (out_dir / "index.html").is_file()
    assert (out_dir / "contact_sheet.png").is_file()
    assert (out_dir / "thumbs" / manifest_rows[0]["output_file"]).is_file()

    report = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "threshold_ladder" in report
    assert "histogram_anchor_grid" in report
    assert "sweep.sqlite" in report


def test_cli_cleans_previous_generated_runs_by_default(tmp_path: Path) -> None:
    gradient = np.tile(np.arange(80, dtype=np.uint8), (48, 1))
    rgb = np.dstack([gradient, gradient, gradient])
    input_path = tmp_path / "gradient.png"
    Image.fromarray(rgb, mode="RGB").save(input_path)
    out_dir = tmp_path / "runs"

    base_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--input",
        str(input_path),
        "--out",
        str(out_dir),
        "--seed",
        "5",
        "--threshold-steps",
        "8",
        "--no-report",
    ]
    subprocess.run([*base_cmd, "--n", "10"], check=True, capture_output=True, text=True)
    subprocess.run([*base_cmd, "--n", "6"], check=True, capture_output=True, text=True)

    with sqlite3.connect(out_dir / "sweep.sqlite") as conn:
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    assert count == 6
    assert len(list(out_dir.glob("sweep_*.png"))) == 6
