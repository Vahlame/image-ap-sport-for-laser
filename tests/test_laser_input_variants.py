from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "laser_input_variants.py"


def test_input_variants_cli_writes_ranked_outputs(tmp_path: Path) -> None:
    x = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (72, 1))
    y = np.tile(np.linspace(255, 0, 72, dtype=np.uint8)[:, None], (1, 96))
    rgb = np.dstack([x, y, ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)])
    input_path = tmp_path / "input.png"
    target_path = tmp_path / "target.png"
    Image.fromarray(rgb, mode="RGB").save(input_path)
    Image.fromarray(255 - rgb, mode="RGB").save(target_path)

    out_dir = tmp_path / "variants"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--target",
            str(target_path),
            "--out",
            str(out_dir),
            "--mode",
            "quick",
            "--limit",
            "8",
            "--score-max-side",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Mejor pre-score" in result.stdout
    assert (out_dir / "index.html").is_file()
    assert (out_dir / "variants.sqlite").is_file()
    assert (out_dir / "variants_manifest.jsonl").is_file()
    assert len(list(out_dir.glob("variant_*.png"))) == 8
    assert len(list((out_dir / "thumbs").glob("variant_*.png"))) == 8

    with sqlite3.connect(out_dir / "variants.sqlite") as conn:
        count, best_score = conn.execute("SELECT COUNT(*), MIN(pre_score) FROM variants").fetchone()
        sources = {row[0] for row in conn.execute("SELECT DISTINCT source FROM variants")}

    assert count == 8
    assert best_score >= 0
    assert "baseline" in sources
