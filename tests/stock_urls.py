"""
Public stock image URLs for automated tests.

Sources (verify before changing URLs):
- **Lorem Picsum** (https://picsum.photos/) — photos from Unsplash, free to use
  under the Unsplash License (https://unsplash.com/license).
- **Wikimedia Commons** — individual files under CC or public domain; each
  entry documents the license link.

Do not use paid stock APIs or hotlinks without a license.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StockImageCase:
    id: str
    url: str
    source: str
    license_name: str
    license_url: str
    expect_min_w: int
    expect_min_h: int


# Stable Picsum URLs: /id/{id}/{w}/{h}
PICSUM_CASES: tuple[StockImageCase, ...] = (
    StockImageCase(
        id="picsum_237_800_600",
        url="https://picsum.photos/id/237/800/600",
        source="Lorem Picsum (Unsplash)",
        license_name="Unsplash License",
        license_url="https://unsplash.com/license",
        expect_min_w=800,
        expect_min_h=600,
    ),
    StockImageCase(
        id="picsum_24_640_480",
        url="https://picsum.photos/id/24/640/480",
        source="Lorem Picsum (Unsplash)",
        license_name="Unsplash License",
        license_url="https://unsplash.com/license",
        expect_min_w=640,
        expect_min_h=480,
    ),
    StockImageCase(
        id="picsum_1003_1280_720",
        url="https://picsum.photos/id/1003/1280/720",
        source="Lorem Picsum (Unsplash)",
        license_name="Unsplash License",
        license_url="https://unsplash.com/license",
        expect_min_w=1280,
        expect_min_h=720,
    ),
    StockImageCase(
        id="picsum_1011_portrait",
        url="https://picsum.photos/id/1011/600/900",
        source="Lorem Picsum (Unsplash)",
        license_name="Unsplash License",
        license_url="https://unsplash.com/license",
        expect_min_w=600,
        expect_min_h=900,
    ),
)

# Wikimedia Commons — NASA / PD government work (verify page if URL changes)
WIKIMEDIA_CASES: tuple[StockImageCase, ...] = (
    StockImageCase(
        id="wikimedia_earth_full",
        url=(
            "https://upload.wikimedia.org/wikipedia/commons/9/97/"
            "The_Earth_seen_from_Apollo_17.jpg"
        ),
        source="Wikimedia Commons — NASA",
        license_name="Public domain (NASA)",
        license_url="https://commons.wikimedia.org/wiki/File:The_Earth_seen_from_Apollo_17.jpg",
        expect_min_w=500,
        expect_min_h=500,
    ),
    StockImageCase(
        id="wikimedia_jupiter_nasa",
        url=(
            "https://upload.wikimedia.org/wikipedia/commons/2/2b/"
            "Jupiter_and_its_shrunken_Great_Red_Spot.jpg"
        ),
        source="Wikimedia Commons — NASA / JPL-Caltech",
        license_name="Public domain",
        license_url="https://commons.wikimedia.org/wiki/File:Jupiter_and_its_shrunken_Great_Red_Spot.jpg",
        expect_min_w=500,
        expect_min_h=400,
    ),
)

ALL_STOCK_CASES: tuple[StockImageCase, ...] = PICSUM_CASES + WIKIMEDIA_CASES
