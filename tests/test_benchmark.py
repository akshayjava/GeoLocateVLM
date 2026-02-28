"""
Unit tests for src/benchmark.py.

Model loading is mocked so tests run without a GPU or network access.

Because benchmark.py lazily imports GeoLocator (from src.inference import
GeoLocator inside run_benchmark), and src.inference requires torch which is not
available in this environment, we inject a fake src.inference module via
patch.dict(sys.modules, ...) so the lazy import resolves to our mock without
ever importing the real module.
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from src.benchmark import run_benchmark, validate_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, rows, name="bench.csv"):
    path = str(tmp_path / name)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_image(tmp_path, name="img.jpg"):
    path = str(tmp_path / name)
    Image.new("RGB", (4, 4)).save(path)
    return path


def _inference_module_mock(predict_return="48.8584, 2.2945"):
    """
    Return a fake src.inference module whose GeoLocator() returns a mock
    with predict() returning predict_return.
    """
    locator_instance = MagicMock()
    locator_instance.predict.return_value = predict_return
    mod = MagicMock()
    mod.GeoLocator.return_value = locator_instance
    return mod


# ---------------------------------------------------------------------------
# validate_csv
# ---------------------------------------------------------------------------

class TestValidateCsv:
    def test_valid_csv_passes(self):
        df = pd.DataFrame({
            "image_path": ["a.jpg"],
            "latitude": [48.0],
            "longitude": [2.0],
        })
        validate_csv(df, "custom")  # must not raise

    def test_missing_image_path_column_raises(self):
        df = pd.DataFrame({"latitude": [0.0], "longitude": [0.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv(df, "custom")

    def test_missing_latitude_column_raises(self):
        df = pd.DataFrame({"image_path": ["a.jpg"], "longitude": [0.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv(df, "custom")

    def test_missing_longitude_column_raises(self):
        df = pd.DataFrame({"image_path": ["a.jpg"], "latitude": [0.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv(df, "custom")

    def test_latitude_above_90_raises(self):
        df = pd.DataFrame({"image_path": ["a.jpg"], "latitude": [91.0], "longitude": [0.0]})
        with pytest.raises(ValueError, match="latitude"):
            validate_csv(df, "custom")

    def test_latitude_below_minus_90_raises(self):
        df = pd.DataFrame({"image_path": ["a.jpg"], "latitude": [-91.0], "longitude": [0.0]})
        with pytest.raises(ValueError, match="latitude"):
            validate_csv(df, "custom")

    def test_longitude_above_180_raises(self):
        df = pd.DataFrame({"image_path": ["a.jpg"], "latitude": [0.0], "longitude": [181.0]})
        with pytest.raises(ValueError, match="longitude"):
            validate_csv(df, "custom")

    def test_longitude_below_minus_180_raises(self):
        df = pd.DataFrame({"image_path": ["a.jpg"], "latitude": [0.0], "longitude": [-181.0]})
        with pytest.raises(ValueError, match="longitude"):
            validate_csv(df, "custom")

    def test_boundary_values_pass(self):
        df = pd.DataFrame({
            "image_path": ["a.jpg", "b.jpg"],
            "latitude": [-90.0, 90.0],
            "longitude": [-180.0, 180.0],
        })
        validate_csv(df, "custom")  # must not raise


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def _run(self, csv, predict_return="48.8584, 2.2945", dataset="custom",
             model_path="models/fake", output_path=None):
        """Run benchmark with a fully mocked src.inference module."""
        mock_mod = _inference_module_mock(predict_return)
        with patch.dict(sys.modules, {"src.inference": mock_mod}):
            return run_benchmark(csv, model_path, dataset, output_path)

    def test_missing_image_counted_in_report(self, tmp_path):
        csv = _write_csv(tmp_path, [{
            "image_path": str(tmp_path / "nonexistent.jpg"),
            "latitude": 0.0,
            "longitude": 0.0,
        }])
        report = self._run(csv)
        assert report["n_missing_images"] == 1

    def test_perfect_prediction_acc_1km_is_one(self, tmp_path):
        img = _write_image(tmp_path)
        csv = _write_csv(tmp_path, [{
            "image_path": img,
            "latitude": 48.8584,
            "longitude": 2.2945,
        }])
        report = self._run(csv, predict_return="48.8584, 2.2945")
        assert report["metrics"]["acc_@1km"] == 1.0

    def test_unparseable_prediction_counted_as_failure(self, tmp_path):
        img = _write_image(tmp_path)
        csv = _write_csv(tmp_path, [{
            "image_path": img,
            "latitude": 48.0,
            "longitude": 2.0,
        }])
        report = self._run(csv, predict_return="Unknown location")
        assert report["metrics"]["parse_failure_rate"] == 1.0
        assert report["metrics"]["acc_@1km"] == 0.0

    def test_report_json_written_to_disk(self, tmp_path):
        img = _write_image(tmp_path)
        csv = _write_csv(tmp_path, [{
            "image_path": img,
            "latitude": 48.8584,
            "longitude": 2.2945,
        }])
        out = str(tmp_path / "report.json")
        self._run(csv, predict_return="48.8584, 2.2945", output_path=out)

        assert os.path.exists(out)
        with open(out) as fh:
            saved = json.load(fh)
        assert "metrics" in saved
        assert saved["dataset"] == "custom"
        assert saved["n_samples"] == 1

    def test_report_contains_required_fields(self, tmp_path):
        img = _write_image(tmp_path)
        csv = _write_csv(tmp_path, [{"image_path": img, "latitude": 0.0, "longitude": 0.0}])
        report = self._run(csv, dataset="im2gps3k", predict_return="0, 0")
        for key in ["dataset", "csv", "model_path", "n_samples",
                    "n_missing_images", "elapsed_seconds", "metrics"]:
            assert key in report

    def test_n_samples_matches_csv_length(self, tmp_path):
        rows = [
            {"image_path": _write_image(tmp_path, f"img{i}.jpg"),
             "latitude": float(i), "longitude": float(i)}
            for i in range(5)
        ]
        csv = _write_csv(tmp_path, rows)
        report = self._run(csv, predict_return="0, 0")
        assert report["n_samples"] == 5

    def test_dataset_label_in_report(self, tmp_path):
        img = _write_image(tmp_path)
        csv = _write_csv(tmp_path, [{"image_path": img, "latitude": 0.0, "longitude": 0.0}])
        report = self._run(csv, dataset="yfcc4k", predict_return="0, 0")
        assert report["dataset"] == "yfcc4k"

    def test_mixed_good_and_missing_images(self, tmp_path):
        img = _write_image(tmp_path)
        rows = [
            {"image_path": img, "latitude": 0.0, "longitude": 0.0},
            {"image_path": str(tmp_path / "ghost.jpg"), "latitude": 0.0, "longitude": 0.0},
        ]
        csv = _write_csv(tmp_path, rows)
        report = self._run(csv, predict_return="0, 0")
        assert report["n_missing_images"] == 1
        assert report["n_samples"] == 2
