"""Streamlit: vista simple del historial en runs/_meta/history.sqlite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


def main() -> None:
    if st is None:
        print(
            "[DASHBOARD] instalar dependencias: pip install -e \".[dashboard]\"",
            flush=True,
        )
        return
    repo = Path(__file__).resolve().parents[1]
    dbp = repo / "runs" / "_meta" / "history.sqlite"
    st.set_page_config(page_title="Laser match history", layout="wide")
    st.title("Historial de experimentos (meta)")
    if not dbp.is_file():
        st.warning(f"No existe {dbp}")
        return
    con = sqlite3.connect(str(dbp))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, started_at, finished_at, best_score, best_pixel_error, preprocess_mode, "
        "score_version, sampling, n_evaluated, input_path, target_path FROM experiments ORDER BY id DESC"
    ).fetchall()
    con.close()
    data = [dict(r) for r in rows]
    if data:
        ids = [int(r["id"]) for r in reversed(data)]
        scores = [float(r["best_score"]) for r in reversed(data)]
        st.line_chart({"id": ids, "best_score": scores})
    st.dataframe(data, use_container_width=True)


if __name__ == "__main__":
    main()
