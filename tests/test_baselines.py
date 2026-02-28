"""
Unit tests for src/baselines.py (Phase 4.5).

All tests run without a GPU or network connection.
"""
import json
import os

import pandas as pd
import pytest

from src.baselines import (
    RandomBaseline,
    MeanBaseline,
    GeoPriorBaseline,
    BASELINES,
    run_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, rows, name="bench.csv"):
    path = str(tmp_path / name)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sample_csv(tmp_path, n=5):
    rows = [
        {"latitude": float(i * 10 - 20), "longitude": float(i * 20 - 40)}
        for i in range(n)
    ]
    return _write_csv(tmp_path, rows), rows


# ---------------------------------------------------------------------------
# RandomBaseline
# ---------------------------------------------------------------------------

class TestRandomBaseline:
    def test_returns_two_floats(self):
        lat, lon = RandomBaseline().predict()
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_lat_in_valid_range(self):
        b = RandomBaseline()
        for _ in range(50):
            lat, _ = b.predict()
            assert -90.0 <= lat <= 90.0

    def test_lon_in_valid_range(self):
        b = RandomBaseline()
        for _ in range(50):
            _, lon = b.predict()
            assert -180.0 <= lon <= 180.0

    def test_not_always_same_value(self):
        b = RandomBaseline()
        predictions = {b.predict() for _ in range(10)}
        # With 10 random samples from a continuous distribution, collisions
        # are astronomically unlikely.
        assert len(predictions) > 1

    def test_image_argument_ignored(self):
        b = RandomBaseline()
        lat, lon = b.predict(image="anything")
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# MeanBaseline
# ---------------------------------------------------------------------------

class TestMeanBaseline:
    def test_default_returns_world_centroid(self):
        b = MeanBaseline()
        lat, lon = b.predict()
        assert lat == pytest.approx(18.0)
        assert lon == pytest.approx(28.0)

    def test_always_returns_same_value(self):
        b = MeanBaseline()
        assert b.predict() == b.predict()

    def test_uses_mean_of_train_csv(self, tmp_path):
        csv = _write_csv(tmp_path, [
            {"latitude": 10.0, "longitude": 20.0},
            {"latitude": 30.0, "longitude": 40.0},
        ])
        b = MeanBaseline(train_csv=csv)
        lat, lon = b.predict()
        assert lat == pytest.approx(20.0)
        assert lon == pytest.approx(30.0)

    def test_image_argument_ignored(self):
        b = MeanBaseline()
        lat, lon = b.predict(image=object())
        assert isinstance(lat, float)
        assert isinstance(lon, float)


# ---------------------------------------------------------------------------
# GeoPriorBaseline
# ---------------------------------------------------------------------------

class TestGeoPriorBaseline:
    def test_returns_two_floats(self):
        lat, lon = GeoPriorBaseline().predict()
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_lat_in_valid_range(self):
        b = GeoPriorBaseline()
        for _ in range(50):
            lat, _ = b.predict()
            assert -90.0 <= lat <= 90.0

    def test_lon_in_valid_range(self):
        b = GeoPriorBaseline()
        for _ in range(50):
            _, lon = b.predict()
            assert -180.0 <= lon <= 180.0

    def test_seeded_instance_is_reproducible(self):
        b1 = GeoPriorBaseline(seed=7)
        b2 = GeoPriorBaseline(seed=7)
        results1 = [b1.predict() for _ in range(20)]
        results2 = [b2.predict() for _ in range(20)]
        assert results1 == results2

    def test_different_seeds_produce_different_sequences(self):
        b1 = GeoPriorBaseline(seed=1)
        b2 = GeoPriorBaseline(seed=2)
        r1 = [b1.predict() for _ in range(10)]
        r2 = [b2.predict() for _ in range(10)]
        assert r1 != r2


# ---------------------------------------------------------------------------
# BASELINES registry
# ---------------------------------------------------------------------------

class TestBaselinesRegistry:
    def test_all_expected_keys_present(self):
        assert set(BASELINES.keys()) == {"random", "mean", "geo_prior"}

    def test_values_are_classes(self):
        for cls in BASELINES.values():
            assert callable(cls)


# ---------------------------------------------------------------------------
# run_baseline
# ---------------------------------------------------------------------------

class TestRunBaseline:
    def test_random_baseline_returns_report(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        report = run_baseline("random", csv, seed=0)
        assert report["baseline"] == "random"
        assert report["n_samples"] == 5
        assert "metrics" in report

    def test_mean_baseline_returns_report(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        report = run_baseline("mean", csv)
        assert report["baseline"] == "mean"
        assert "metrics" in report

    def test_geo_prior_baseline_returns_report(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        report = run_baseline("geo_prior", csv, seed=42)
        assert report["baseline"] == "geo_prior"
        assert "metrics" in report

    def test_report_metrics_have_required_keys(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        report = run_baseline("random", csv, seed=0)
        m = report["metrics"]
        for key in ["acc_@1km", "acc_@25km", "acc_@200km", "acc_@750km",
                    "acc_@2500km", "median_error_km", "mean_error_km"]:
            assert key in m

    def test_json_report_written_to_disk(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        out = str(tmp_path / "report.json")
        run_baseline("random", csv, output_path=out, seed=0)
        assert os.path.exists(out)
        with open(out) as fh:
            saved = json.load(fh)
        assert saved["baseline"] == "random"

    def test_unknown_baseline_raises_value_error(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        with pytest.raises(ValueError, match="Unknown baseline"):
            run_baseline("nonexistent", csv)

    def test_missing_latitude_column_raises(self, tmp_path):
        csv = _write_csv(tmp_path, [{"longitude": 0.0}])
        with pytest.raises(ValueError, match="latitude"):
            run_baseline("random", csv)

    def test_missing_longitude_column_raises(self, tmp_path):
        csv = _write_csv(tmp_path, [{"latitude": 0.0}])
        with pytest.raises(ValueError, match="longitude"):
            run_baseline("random", csv)

    def test_mean_baseline_with_train_csv(self, tmp_path):
        bench_csv, _ = _sample_csv(tmp_path)
        train_csv = _write_csv(tmp_path, [
            {"latitude": 50.0, "longitude": 10.0},
        ], name="train.csv")
        report = run_baseline("mean", bench_csv, train_csv=train_csv)
        # All predictions should be at exactly (50.0, 10.0)
        assert report["metrics"]["parse_failure_rate"] == 0.0

    def test_n_samples_matches_csv_row_count(self, tmp_path):
        csv, rows = _sample_csv(tmp_path, n=8)
        report = run_baseline("random", csv, seed=0)
        assert report["n_samples"] == 8

    def test_seeded_random_baseline_is_reproducible(self, tmp_path):
        csv, _ = _sample_csv(tmp_path)
        r1 = run_baseline("random", csv, seed=99)
        r2 = run_baseline("random", csv, seed=99)
        assert r1["metrics"]["median_error_km"] == pytest.approx(
            r2["metrics"]["median_error_km"]
        )
