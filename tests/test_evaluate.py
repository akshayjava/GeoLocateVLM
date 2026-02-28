"""
Unit tests for src/evaluate.py.

Covers parse_coordinates and calculate_metrics without loading any model.
"""
import pytest
from src.evaluate import parse_coordinates, calculate_metrics


# ---------------------------------------------------------------------------
# parse_coordinates
# ---------------------------------------------------------------------------

class TestParseCoordinates:
    def test_standard_decimal(self):
        assert parse_coordinates("48.8584, 2.2945") == (48.8584, 2.2945)

    def test_integer_coordinates(self):
        # Regression: old regex required a decimal point
        assert parse_coordinates("48, 2") == (48.0, 2.0)

    def test_negative_lat(self):
        assert parse_coordinates("-33.87, 151.21") == (-33.87, 151.21)

    def test_both_negative(self):
        assert parse_coordinates("-33.87, -70.65") == (-33.87, -70.65)

    def test_zero_zero(self):
        assert parse_coordinates("0, 0") == (0.0, 0.0)

    def test_embedded_in_sentence(self):
        result = parse_coordinates("Taken in Paris, France at 48.8584, 2.2945 last summer.")
        assert result == (48.8584, 2.2945)

    def test_extra_whitespace(self):
        assert parse_coordinates("48.8584,   2.2945") == (48.8584, 2.2945)

    def test_city_country_only_returns_none(self):
        # "Paris, France" should NOT be parsed as coordinates
        # "Paris" is not a number, so the regex won't match
        result = parse_coordinates("Paris, France")
        assert result is None

    def test_empty_string_returns_none(self):
        assert parse_coordinates("") is None

    def test_single_number_returns_none(self):
        assert parse_coordinates("48.8584") is None

    def test_first_match_returned(self):
        # When multiple coordinate pairs appear, the first is returned
        result = parse_coordinates("from 10.0, 20.0 to 30.0, 40.0")
        assert result == (10.0, 20.0)


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def test_empty_results_returns_empty_dict(self):
        assert calculate_metrics([]) == {}

    def test_all_required_keys_present(self):
        results = [{"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 0.0, "pred_lon": 0.0}]
        metrics = calculate_metrics(results)
        expected = {
            "acc_@1km", "acc_@25km", "acc_@200km", "acc_@750km", "acc_@2500km",
            "median_error_km", "mean_error_km", "parse_failure_rate",
        }
        assert expected.issubset(metrics.keys())

    def test_perfect_prediction_all_acc_one(self):
        results = [{"true_lat": 48.8584, "true_lon": 2.2945,
                    "pred_lat": 48.8584, "pred_lon": 2.2945}]
        m = calculate_metrics(results)
        assert m["acc_@1km"] == 1.0
        assert m["acc_@2500km"] == 1.0
        assert m["median_error_km"] == pytest.approx(0.0, abs=0.01)
        assert m["parse_failure_rate"] == 0.0

    def test_all_failed_predictions(self):
        results = [{"true_lat": 48.8, "true_lon": 2.3, "pred_lat": None, "pred_lon": None}]
        m = calculate_metrics(results)
        assert m["acc_@1km"] == 0.0
        assert m["acc_@2500km"] == 0.0
        assert m["parse_failure_rate"] == 1.0
        assert m["median_error_km"] == float("inf")
        assert m["mean_error_km"] == float("inf")

    def test_threshold_accuracy_boundary(self):
        # ~11 km apart (0.1 degree latitude ≈ 11 km)
        # ~1111 km apart (10 degrees latitude ≈ 1111 km)
        results = [
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 0.1, "pred_lon": 0.0},
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 10.0, "pred_lon": 0.0},
        ]
        m = calculate_metrics(results)
        assert m["acc_@1km"] == 0.0      # neither within 1 km
        assert m["acc_@25km"] == 0.5     # only first within 25 km
        assert m["acc_@2500km"] == 1.0   # both within 2500 km

    def test_parse_failure_rate_partial(self):
        results = [
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 0.0, "pred_lon": 0.0},
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": None, "pred_lon": None},
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": None, "pred_lon": None},
        ]
        m = calculate_metrics(results)
        assert m["parse_failure_rate"] == pytest.approx(2 / 3)

    def test_inf_excluded_from_mean_and_median(self):
        # One good prediction (0 km error) and one failed parse
        results = [
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 0.0, "pred_lon": 0.0},
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": None, "pred_lon": None},
        ]
        m = calculate_metrics(results)
        assert m["median_error_km"] != float("inf")
        assert m["mean_error_km"] != float("inf")
        assert m["median_error_km"] == pytest.approx(0.0, abs=0.01)

    def test_accuracy_values_in_zero_one_range(self):
        results = [
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 0.0, "pred_lon": 0.0},
            {"true_lat": 0.0, "true_lon": 0.0, "pred_lat": 90.0, "pred_lon": 0.0},
        ]
        m = calculate_metrics(results)
        for key in ["acc_@1km", "acc_@25km", "acc_@200km", "acc_@750km", "acc_@2500km"]:
            assert 0.0 <= m[key] <= 1.0
