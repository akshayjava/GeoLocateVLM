"""
Im2GPS3k dataset loader (Phase 4.3).

The Im2GPS3k test set contains 2,997 geotagged Flickr images and is the
standard benchmark for geo-localisation models.

Paper  : Vo et al., "Revisiting Im2GPS in the Deep Learning Era", ICCV 2017
         https://arxiv.org/abs/1705.01061
Download: https://graphics.cs.cmu.edu/projects/im2gps/

Expected directory layout
--------------------------
data/benchmarks/im2gps3k/
├── *.csv                  # GPS metadata (any .csv in the directory)
│                          # Original: im2gps3k_places365.csv
│                          # Required columns (case-insensitive):
│                          #   image_file / IMG_ID / filename
│                          #   lat / LAT / latitude
│                          #   lon / LON / lng / longitude
└── images/                # Downloaded JPEG images

Building the CSV
-----------------
The original GPS metadata from the Im2GPS3k repository uses columns
IMG_ID, LAT, LON.  A normalised variant with image_file, lat, lon is
also accepted.

Usage
-----
    from src.benchmarks.im2gps3k import load_im2gps3k
    from src.evaluate import calculate_metrics

    results = []
    for image, lat, lon in load_im2gps3k("data/benchmarks/im2gps3k"):
        pred = locator.predict(image)
        ...
"""

import warnings
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image

EXPECTED_DIR = "data/benchmarks/im2gps3k"

# Known column-name mappings  (our_key -> csv_column_name)
_COLUMN_CANDIDATES = {
    "image_file": ["img_id", "image_file", "filename", "file", "image"],
    "lat":        ["lat", "latitude"],
    "lon":        ["lon", "lng", "longitude"],
}


def load_im2gps3k(
    data_dir: str = EXPECTED_DIR,
    images_subdir: str = "images",
    max_samples: int | None = None,
) -> Iterator[tuple[Image.Image, float, float]]:
    """
    Yield (PIL.Image, latitude, longitude) for each image in Im2GPS3k.

    Images that cannot be found or opened are silently skipped; a warning
    is issued at the end if any images were missing.

    Parameters
    ----------
    data_dir      : Root directory containing the CSV and images/ subdirectory.
    images_subdir : Name of the subdirectory that holds the JPEG files.
    max_samples   : Stop after yielding this many samples (None = all).

    Yields
    ------
    PIL.Image (RGB mode), latitude (float), longitude (float)

    Raises
    ------
    FileNotFoundError
        If data_dir does not exist or no CSV file is found inside it.
    ValueError
        If the CSV does not contain recognisable lat/lon/image columns.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Im2GPS3k data directory not found: {data_dir}\n"
            "To obtain the dataset:\n"
            "  1. Visit https://graphics.cs.cmu.edu/projects/im2gps/\n"
            "  2. Download the test images and GPS metadata CSV.\n"
            f"  3. Place them in {data_dir}/ with an images/ subdirectory."
        )

    csv_candidates = sorted(data_path.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV file found in {data_dir}. "
            "Expected a file such as im2gps3k_places365.csv with columns "
            "IMG_ID (or image_file), LAT (or lat), LON (or lon)."
        )

    csv_path = csv_candidates[0]
    df = pd.read_csv(csv_path)

    image_col, lat_col, lon_col = _detect_columns(df, csv_path)
    images_dir = data_path / images_subdir

    n_yielded = 0
    n_missing = 0

    for _, row in df.iterrows():
        if max_samples is not None and n_yielded >= max_samples:
            break

        img_name = str(row[image_col])
        img_path = images_dir / img_name
        if not img_path.suffix:
            img_path = img_path.with_suffix(".jpg")

        if not img_path.exists():
            n_missing += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            yield image, float(row[lat_col]), float(row[lon_col])
            n_yielded += 1
        except Exception:
            n_missing += 1
            continue

    if n_missing:
        warnings.warn(
            f"Im2GPS3k: skipped {n_missing} missing or unreadable image(s) "
            f"(yielded {n_yielded} samples).",
            UserWarning,
            stacklevel=2,
        )


def _detect_columns(df: pd.DataFrame, csv_path) -> tuple[str, str, str]:
    """
    Identify the image-file, latitude, and longitude columns in *df*.

    Matching is case-insensitive.  Returns (image_col, lat_col, lon_col).
    Raises ValueError if any of the three required roles cannot be mapped.
    """
    lower_to_actual = {c.lower(): c for c in df.columns}
    resolved = {}

    for role, candidates in _COLUMN_CANDIDATES.items():
        for name in candidates:
            if name in lower_to_actual:
                resolved[role] = lower_to_actual[name]
                break

    missing_roles = [r for r in _COLUMN_CANDIDATES if r not in resolved]
    if missing_roles:
        raise ValueError(
            f"Cannot find required columns {missing_roles} in {csv_path}. "
            f"Available columns: {list(df.columns)}. "
            "Expected something like: IMG_ID/image_file, LAT/lat, LON/lon."
        )

    return resolved["image_file"], resolved["lat"], resolved["lon"]
