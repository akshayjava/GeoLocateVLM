"""
YFCC-val dataset loader (Phase 4.4).

The YFCC100M OpenAI subset is streamed directly from HuggingFace — no bulk
download is needed.  GPS coordinates are embedded in the 'tags' field of each
sample as semicolon-separated geo: tags.

Dataset  : dalle-mini/YFCC100M_OpenAI_subset
HF URL   : https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset

Tag format (example)
---------------------
    geotagged;geo:lat=48.8584;geo:lon=2.2945;...

Only samples that contain both geo:lat and geo:lon tags are yielded;
all others are silently skipped.

Offline variant
---------------
For reproducible evaluation without an internet connection, use
load_yfcc_from_csv() with a pre-downloaded CSV:

    image_path, latitude, longitude
    /path/to/img.jpg, 48.8584, 2.2945

Usage
-----
    # Streaming (requires internet + `datasets` package)
    from src.benchmarks.yfcc_val import load_yfcc_val

    for image, lat, lon in load_yfcc_val(max_samples=1000):
        pred = locator.predict(image)
        ...

    # Offline
    from src.benchmarks.yfcc_val import load_yfcc_from_csv

    for image, lat, lon in load_yfcc_from_csv("data/benchmarks/yfcc_val.csv"):
        ...
"""

import re
import warnings
from pathlib import Path
from typing import Iterator

from PIL import Image

# ---------------------------------------------------------------------------
# Coordinate parsing from YFCC tag strings
# ---------------------------------------------------------------------------

_LAT_RE = re.compile(r"geo:lat=(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_LON_RE = re.compile(r"geo:lon=(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_yfcc_coords(tags: str) -> tuple[float, float] | None:
    """
    Extract (latitude, longitude) from a YFCC100M tag string.

    Parameters
    ----------
    tags : Semicolon-separated tag string, e.g.
           "geotagged;geo:lat=48.8584;geo:lon=2.2945;landscape"

    Returns
    -------
    (lat, lon) as floats, or None if coordinates are missing or out of range.
    """
    if not tags:
        return None

    lat_m = _LAT_RE.search(tags)
    lon_m = _LON_RE.search(tags)
    if not lat_m or not lon_m:
        return None

    lat = float(lat_m.group(1))
    lon = float(lon_m.group(1))

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    return lat, lon


# ---------------------------------------------------------------------------
# Streaming loader (HuggingFace)
# ---------------------------------------------------------------------------

def load_yfcc_val(
    split: str = "validation",
    max_samples: int | None = 4000,
    min_image_size: int = 32,
) -> Iterator[tuple[Image.Image, float, float]]:
    """
    Stream geotagged samples from the YFCC100M OpenAI subset on HuggingFace.

    Requires an active internet connection and the ``datasets`` package.
    Images are downloaded on-the-fly; no local storage is needed beyond a
    small in-memory buffer.

    Parameters
    ----------
    split         : HuggingFace dataset split to use (default: 'validation').
    max_samples   : Maximum number of *geotagged* samples to yield.
                    Set to None to stream the entire split (can be very slow).
    min_image_size: Skip images whose width or height is below this value
                    (guards against corrupted thumbnail entries).

    Yields
    ------
    PIL.Image (RGB mode), latitude (float), longitude (float)

    Raises
    ------
    ImportError  If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for YFCC streaming. "
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset(
        "dalle-mini/YFCC100M_OpenAI_subset",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    n_yielded = 0
    n_skipped = 0

    for sample in ds:
        if max_samples is not None and n_yielded >= max_samples:
            break

        # Extract GPS coordinates from the tag field
        tags = sample.get("tags") or sample.get("description") or ""
        coords = parse_yfcc_coords(str(tags))
        if coords is None:
            n_skipped += 1
            continue

        raw_image = sample.get("image")
        if raw_image is None:
            n_skipped += 1
            continue

        try:
            if not isinstance(raw_image, Image.Image):
                raw_image = Image.fromarray(raw_image)
            image = raw_image.convert("RGB")
            if image.width < min_image_size or image.height < min_image_size:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        yield image, coords[0], coords[1]
        n_yielded += 1

    if n_skipped:
        warnings.warn(
            f"YFCC-val: skipped {n_skipped} sample(s) with missing/invalid "
            f"GPS data or images (yielded {n_yielded}).",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Offline CSV loader
# ---------------------------------------------------------------------------

def load_yfcc_from_csv(
    csv_path: str,
    max_samples: int | None = None,
) -> Iterator[tuple[Image.Image, float, float]]:
    """
    Load YFCC-val images from a pre-downloaded local CSV.

    This is the offline, reproducible equivalent of :func:`load_yfcc_val`.

    CSV format
    ----------
    Required columns: ``image_path``, ``latitude``, ``longitude``

    Parameters
    ----------
    csv_path    : Path to the CSV file.
    max_samples : Stop after yielding this many samples (None = all).

    Yields
    ------
    PIL.Image (RGB mode), latitude (float), longitude (float)

    Raises
    ------
    ValueError  If the CSV is missing required columns.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"image_path", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    n_yielded = 0
    for _, row in df.iterrows():
        if max_samples is not None and n_yielded >= max_samples:
            break

        img_path = Path(row["image_path"])
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            yield image, float(row["latitude"]), float(row["longitude"])
            n_yielded += 1
        except Exception:
            continue
