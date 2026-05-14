"""Shared fixtures: download public stock images once per pytest session."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pytest

from tests.stock_urls import StockImageCase


@pytest.fixture(scope="session")
def stock_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("stock_downloads")


@pytest.fixture(scope="session")
def download_stock(
    stock_cache_dir: Path,
) -> Callable[[StockImageCase], Path]:
    def _download(case: StockImageCase) -> Path:
        suffix = ".jpg"
        if ".png" in case.url.lower():
            suffix = ".png"
        path = stock_cache_dir / f"{case.id}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
        req = urllib.request.Request(
            case.url,
            headers={
                "User-Agent": (
                    "image-ap-sport-for-laser-tests/0.1 "
                    "(automated pytest stock downloads; https://github.com/)"
                ),
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            pytest.skip(f"network download failed for {case.id}: {e}")
        if not data:
            pytest.skip(f"empty response for {case.id}")
        path.write_bytes(data)
        return path

    return _download
