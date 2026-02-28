"""
Unit tests for src/benchmarks/im2gps3k.py and src/benchmarks/yfcc_val.py
(Phase 4.3 and 4.4).

All tests run without a GPU or network connection.  The YFCC streaming
loader is tested via the offline CSV variant and the coordinate-parsing
helper; the live HuggingFace streaming path is not exercised in unit tests.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.benchmarks.im2gps3k import load_im2gps3k, _detect_columns
from src.benchmarks.yfcc_val import (
    parse_yfcc_coords,
    load_yfcc_from_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(tmp_path, name="img.jpg", size=(8, 8)):
    path = tmp_path / name
    Image.new("RGB", size).save(str(path))
    return path


def _write_csv(tmp_path, rows, name="meta.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(str(path), index=False)
    return path


# ---------------------------------------------------------------------------
# Im2GPS3k loader
# ---------------------------------------------------------------------------

class TestLoadIm2GPS3k:
    def _setup_dataset(self, tmp_path, rows, csv_name="meta.csv",
                       images_subdir="images", img_names=None):
        """Create a minimal Im2GPS3k directory fixture."""
        images_dir = tmp_path / images_subdir
        images_dir.mkdir(parents=True, exist_ok=True)

        if img_names is None:
            img_names = [str(r.get("image_file") or r.get("IMG_ID") or f"{i}.jpg")
                         for i, r in enumerate(rows)]

        for name in img_names:
            _make_image(images_dir, name)

        _write_csv(tmp_path, rows, csv_name)
        return str(tmp_path)

    # --- happy paths ---

    def test_yields_correct_number_of_samples(self, tmp_path):
        rows = [
            {"image_file": "a.jpg", "lat": 48.8584, "lon": 2.2945},
            {"image_file": "b.jpg", "lat": -33.8688, "lon": 151.2093},
        ]
        data_dir = self._setup_dataset(tmp_path, rows)
        samples = list(load_im2gps3k(data_dir))
        assert len(samples) == 2

    def test_yields_pil_images_rgb(self, tmp_path):
        rows = [{"image_file": "x.jpg", "lat": 0.0, "lon": 0.0}]
        data_dir = self._setup_dataset(tmp_path, rows)
        for image, _, _ in load_im2gps3k(data_dir):
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"

    def test_yields_correct_coordinates(self, tmp_path):
        rows = [{"image_file": "p.jpg", "lat": 48.8584, "lon": 2.2945}]
        data_dir = self._setup_dataset(tmp_path, rows)
        samples = list(load_im2gps3k(data_dir))
        _, lat, lon = samples[0]
        assert lat == pytest.approx(48.8584)
        assert lon == pytest.approx(2.2945)

    def test_accepts_uppercase_column_names(self, tmp_path):
        rows = [{"IMG_ID": "img.jpg", "LAT": 10.0, "LON": 20.0}]
        data_dir = self._setup_dataset(tmp_path, rows)
        samples = list(load_im2gps3k(data_dir))
        assert len(samples) == 1
        _, lat, lon = samples[0]
        assert lat == pytest.approx(10.0)
        assert lon == pytest.approx(20.0)

    def test_max_samples_limits_output(self, tmp_path):
        rows = [
            {"image_file": f"{i}.jpg", "lat": float(i), "lon": float(i)}
            for i in range(5)
        ]
        data_dir = self._setup_dataset(tmp_path, rows)
        samples = list(load_im2gps3k(data_dir, max_samples=2))
        assert len(samples) == 2

    def test_missing_images_are_skipped_with_warning(self, tmp_path):
        # Create only one of the two referenced images
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir, "present.jpg")
        rows = [
            {"image_file": "present.jpg", "lat": 0.0, "lon": 0.0},
            {"image_file": "ghost.jpg",   "lat": 1.0, "lon": 1.0},
        ]
        _write_csv(tmp_path, rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            samples = list(load_im2gps3k(str(tmp_path)))
        assert len(samples) == 1
        assert any("skipped" in str(warning.message).lower() for warning in w)

    # --- error paths ---

    def test_missing_data_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            list(load_im2gps3k(str(tmp_path / "nonexistent")))

    def test_no_csv_raises_file_not_found(self, tmp_path):
        (tmp_path / "images").mkdir()
        with pytest.raises(FileNotFoundError, match="No CSV"):
            list(load_im2gps3k(str(tmp_path)))

    def test_unrecognised_columns_raise_value_error(self, tmp_path):
        rows = [{"foo": "bar.jpg", "x": 0.0, "y": 0.0}]
        data_dir = self._setup_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="Cannot find required columns"):
            list(load_im2gps3k(data_dir))


class TestDetectColumns:
    def test_normalised_lowercase_columns(self):
        df = pd.DataFrame(columns=["image_file", "lat", "lon"])
        img, lat, lon = _detect_columns(df, "test.csv")
        assert img == "image_file"
        assert lat == "lat"
        assert lon == "lon"

    def test_original_uppercase_columns(self):
        df = pd.DataFrame(columns=["IMG_ID", "LAT", "LON"])
        img, lat, lon = _detect_columns(df, "test.csv")
        assert img == "IMG_ID"
        assert lat == "LAT"
        assert lon == "LON"

    def test_latitude_longitude_variants(self):
        df = pd.DataFrame(columns=["filename", "latitude", "longitude"])
        img, lat, lon = _detect_columns(df, "test.csv")
        assert img == "filename"
        assert lat == "latitude"
        assert lon == "longitude"

    def test_missing_columns_raises_value_error(self):
        df = pd.DataFrame(columns=["only_col"])
        with pytest.raises(ValueError, match="Cannot find required columns"):
            _detect_columns(df, "test.csv")


# ---------------------------------------------------------------------------
# YFCC coordinate parsing
# ---------------------------------------------------------------------------

class TestParseYfccCoords:
    def test_standard_geo_tags(self):
        tags = "geotagged;geo:lat=48.8584;geo:lon=2.2945;landscape"
        result = parse_yfcc_coords(tags)
        assert result == pytest.approx((48.8584, 2.2945))

    def test_negative_coordinates(self):
        tags = "geotagged;geo:lat=-33.8688;geo:lon=151.2093"
        result = parse_yfcc_coords(tags)
        assert result == pytest.approx((-33.8688, 151.2093))

    def test_integer_coordinates(self):
        tags = "geo:lat=48;geo:lon=2"
        result = parse_yfcc_coords(tags)
        assert result == (48.0, 2.0)

    def test_case_insensitive(self):
        tags = "GEO:LAT=10.0;GEO:LON=20.0"
        result = parse_yfcc_coords(tags)
        assert result == pytest.approx((10.0, 20.0))

    def test_missing_lat_returns_none(self):
        assert parse_yfcc_coords("geo:lon=2.2945") is None

    def test_missing_lon_returns_none(self):
        assert parse_yfcc_coords("geo:lat=48.8584") is None

    def test_empty_string_returns_none(self):
        assert parse_yfcc_coords("") is None

    def test_none_input_returns_none(self):
        assert parse_yfcc_coords(None) is None

    def test_invalid_lat_out_of_range_returns_none(self):
        assert parse_yfcc_coords("geo:lat=95.0;geo:lon=0.0") is None

    def test_invalid_lon_out_of_range_returns_none(self):
        assert parse_yfcc_coords("geo:lat=0.0;geo:lon=200.0") is None

    def test_no_geo_tags_returns_none(self):
        assert parse_yfcc_coords("landscape;travel;nature") is None

    def test_boundary_values_are_valid(self):
        assert parse_yfcc_coords("geo:lat=90.0;geo:lon=180.0") == (90.0, 180.0)
        assert parse_yfcc_coords("geo:lat=-90.0;geo:lon=-180.0") == (-90.0, -180.0)


# ---------------------------------------------------------------------------
# YFCC offline CSV loader
# ---------------------------------------------------------------------------

class TestLoadYfccFromCsv:
    def test_yields_images_and_coordinates(self, tmp_path):
        img = _make_image(tmp_path, "y.jpg")
        csv = _write_csv(tmp_path, [{
            "image_path": str(img),
            "latitude": 48.8584,
            "longitude": 2.2945,
        }])
        samples = list(load_yfcc_from_csv(str(csv)))
        assert len(samples) == 1
        image, lat, lon = samples[0]
        assert isinstance(image, Image.Image)
        assert lat == pytest.approx(48.8584)
        assert lon == pytest.approx(2.2945)

    def test_missing_images_are_silently_skipped(self, tmp_path):
        csv = _write_csv(tmp_path, [{
            "image_path": str(tmp_path / "nonexistent.jpg"),
            "latitude": 0.0,
            "longitude": 0.0,
        }])
        samples = list(load_yfcc_from_csv(str(csv)))
        assert samples == []

    def test_max_samples_limits_output(self, tmp_path):
        rows = []
        for i in range(4):
            img = _make_image(tmp_path, f"i{i}.jpg")
            rows.append({"image_path": str(img), "latitude": float(i), "longitude": 0.0})
        csv = _write_csv(tmp_path, rows)
        samples = list(load_yfcc_from_csv(str(csv), max_samples=2))
        assert len(samples) == 2

    def test_missing_image_path_column_raises(self, tmp_path):
        csv = _write_csv(tmp_path, [{"latitude": 0.0, "longitude": 0.0}])
        with pytest.raises(ValueError, match="missing required columns"):
            list(load_yfcc_from_csv(str(csv)))

    def test_missing_latitude_column_raises(self, tmp_path):
        csv = _write_csv(tmp_path, [{"image_path": "x.jpg", "longitude": 0.0}])
        with pytest.raises(ValueError, match="missing required columns"):
            list(load_yfcc_from_csv(str(csv)))

    def test_missing_longitude_column_raises(self, tmp_path):
        csv = _write_csv(tmp_path, [{"image_path": "x.jpg", "latitude": 0.0}])
        with pytest.raises(ValueError, match="missing required columns"):
            list(load_yfcc_from_csv(str(csv)))

    def test_yields_rgb_images(self, tmp_path):
        img = _make_image(tmp_path, "rgb.jpg")
        csv = _write_csv(tmp_path, [{"image_path": str(img), "latitude": 0.0, "longitude": 0.0}])
        for image, _, _ in load_yfcc_from_csv(str(csv)):
            assert image.mode == "RGB"
