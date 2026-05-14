"""Tests using publicly licensed stock URLs (see tests/stock_urls.py)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image, ImageStat

from tests.stock_urls import ALL_STOCK_CASES, PICSUM_CASES, StockImageCase, WIKIMEDIA_CASES


@pytest.mark.network
@pytest.mark.parametrize("case", ALL_STOCK_CASES, ids=lambda c: c.id)
def test_stock_download_opens_and_dimensions(
    case: StockImageCase,
    download_stock: Callable[[StockImageCase], Path],
) -> None:
    path = download_stock(case)
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        w, h = im.size
    assert w >= case.expect_min_w - 50, f"{case.id}: width {w} smaller than expected"
    assert h >= case.expect_min_h - 50, f"{case.id}: height {h} smaller than expected"


@pytest.mark.network
@pytest.mark.parametrize("case", ALL_STOCK_CASES, ids=lambda c: c.id)
def test_stock_convert_grayscale_and_thumbnail(
    case: StockImageCase,
    download_stock: Callable[[StockImageCase], Path],
) -> None:
    path = download_stock(case)
    with Image.open(path) as im:
        gray = im.convert("L")
        thumb = gray.copy()
        thumb.thumbnail((256, 256), Image.Resampling.LANCZOS)
    assert thumb.size[0] <= 256 and thumb.size[1] <= 256
    assert thumb.mode == "L"


@pytest.mark.network
@pytest.mark.parametrize("case", ALL_STOCK_CASES, ids=lambda c: c.id)
def test_stock_histogram_has_variance(
    case: StockImageCase,
    download_stock: Callable[[StockImageCase], Path],
) -> None:
    """Flat or single-color images are poor laser subjects; ensure some tonal spread."""
    path = download_stock(case)
    with Image.open(path) as im:
        stat = ImageStat.Stat(im.convert("L"))
    # stddev near 0 means almost solid color
    assert stat.stddev[0] > 2.0, f"{case.id}: image may be flat (stddev={stat.stddev[0]})"


@pytest.mark.network
def test_picsum_batch_all_distinct(
    download_stock: Callable[[StockImageCase], Path],
) -> None:
    hashes = []
    for case in PICSUM_CASES:
        path = download_stock(case)
        hashes.append(path.read_bytes()[:4096])
    assert len(hashes) == len(set(hashes)), "expected distinct image payloads for Picsum cases"


@pytest.mark.network
def test_wikimedia_urls_are_jpeg_or_png(
    download_stock: Callable[[StockImageCase], Path],
) -> None:
    for case in WIKIMEDIA_CASES:
        path = download_stock(case)
        fmt = Image.open(path).format
        assert fmt in ("JPEG", "PNG"), f"{case.id}: unexpected format {fmt}"


def test_stock_case_ids_unique() -> None:
    ids_ = [c.id for c in ALL_STOCK_CASES]
    counts = Counter(ids_)
    dupes = [k for k, v in counts.items() if v > 1]
    assert not dupes, f"duplicate case ids: {dupes}"


def test_stock_urls_are_https() -> None:
    for c in ALL_STOCK_CASES:
        assert c.url.startswith("https://"), c.id
