"""
Baseline geolocation predictors for benchmarking comparison (Phase 4.5).

These baselines provide reference performance levels so that model results
are interpretable without consulting the literature.

Expected approximate median errors
-----------------------------------
Random       ≈ 5 000 km  (absolute floor)
Geo-prior    ≈ 3 500 km  (land-biased random)
Mean         ≈ 4 000 km  (always predicts training-set centroid)

Any useful fine-tuned model should significantly outperform all three.

Usage
-----
    from src.baselines import RandomBaseline, MeanBaseline, GeoPriorBaseline
    from src.evaluate import calculate_metrics

    baseline = RandomBaseline()
    results = []
    for _, true_lat, true_lon in dataset:
        pred_lat, pred_lon = baseline.predict()
        results.append({
            "true_lat": true_lat, "true_lon": true_lon,
            "pred_lat": pred_lat, "pred_lon": pred_lon,
        })
    metrics = calculate_metrics(results)

CLI
---
    python src/baselines.py \\
        --baseline random \\
        --csv data/benchmarks/im2gps3k.csv \\
        --output results/baseline_random.json
"""

import argparse
import json
import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluate import calculate_metrics


# ---------------------------------------------------------------------------
# Baseline predictor classes
# ---------------------------------------------------------------------------

class RandomBaseline:
    """
    Uniformly random latitude and longitude.

    Samples lat from [-90, 90] and lon from [-180, 180] independently.
    Expected median great-circle error ≈ 5 000 km.
    """

    def predict(self, image=None) -> tuple[float, float]:
        """Return a uniformly random (lat, lon) pair."""
        return random.uniform(-90.0, 90.0), random.uniform(-180.0, 180.0)


class MeanBaseline:
    """
    Always predicts the mean GPS coordinates of a reference set.

    When no training CSV is provided, the hard-coded world
    population-weighted centroid (≈ 18°N, 28°E) is used as a reasonable
    stand-in.

    Parameters
    ----------
    train_csv : Optional path to a CSV with 'latitude' and 'longitude'
                columns.  If provided, the mean of those coordinates
                is used as the fixed prediction.
    """

    # Approximate world population-weighted centroid
    _DEFAULT_LAT = 18.0
    _DEFAULT_LON = 28.0

    def __init__(self, train_csv: str | None = None):
        if train_csv:
            df = pd.read_csv(train_csv)
            self._lat = float(df["latitude"].mean())
            self._lon = float(df["longitude"].mean())
        else:
            self._lat = self._DEFAULT_LAT
            self._lon = self._DEFAULT_LON

    def predict(self, image=None) -> tuple[float, float]:
        """Always return the pre-computed mean coordinates."""
        return self._lat, self._lon


class GeoPriorBaseline:
    """
    Samples a uniformly random point from a coarse land-mass grid.

    Approximates the "geo-prior" baseline from the Im2GPS literature by
    restricting predictions to approximate continental bounding boxes.
    Expected median error ≈ 3 500 km — better than fully random because
    it never places predictions in the open ocean.

    Parameters
    ----------
    seed : Optional random seed for reproducibility.
    """

    # Approximate continental bounding boxes
    # Each entry: (min_lat, max_lat, min_lon, max_lon)
    _LAND_BOXES = [
        # North America
        (15.0,  72.0, -168.0,  -52.0),
        # South America
        (-56.0, 13.0,  -81.0,  -34.0),
        # Europe
        (35.0,  71.0,  -10.0,   40.0),
        # Africa
        (-35.0, 37.0,  -18.0,   52.0),
        # Asia (west + central)
        (10.0,  70.0,   25.0,   90.0),
        # Asia (east)
        (10.0,  70.0,   90.0,  145.0),
        # South / Southeast Asia
        (-10.0, 28.0,   65.0,  145.0),
        # Australia / Oceania
        (-44.0, -10.0, 113.0,  154.0),
    ]

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def predict(self, image=None) -> tuple[float, float]:
        """Return a random point sampled from an approximate land-mass grid."""
        box = self._rng.choice(self._LAND_BOXES)
        lat = self._rng.uniform(box[0], box[1])
        lon = self._rng.uniform(box[2], box[3])
        return lat, lon


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES: dict[str, type] = {
    "random":    RandomBaseline,
    "mean":      MeanBaseline,
    "geo_prior": GeoPriorBaseline,
}


# ---------------------------------------------------------------------------
# Bulk evaluation helper
# ---------------------------------------------------------------------------

def run_baseline(
    baseline_name: str,
    csv_path: str,
    output_path: str | None = None,
    seed: int = 42,
    train_csv: str | None = None,
) -> dict:
    """
    Evaluate a named baseline on a benchmark CSV and return a report dict.

    Parameters
    ----------
    baseline_name : One of 'random', 'mean', 'geo_prior'.
    csv_path      : CSV with columns: latitude, longitude
                    (image_path column is optional for baselines).
    output_path   : If given, the JSON report is written to this path.
    seed          : Random seed for reproducibility.
    train_csv     : For the 'mean' baseline: CSV used to compute mean
                    coordinates. Falls back to the world centroid if None.

    Returns
    -------
    dict with keys: baseline, csv, n_samples, elapsed_seconds, metrics
    """
    if baseline_name not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{baseline_name}'. "
            f"Available options: {sorted(BASELINES.keys())}"
        )

    df = pd.read_csv(csv_path)
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            raise ValueError(
                f"CSV '{csv_path}' is missing required column: '{col}'"
            )

    random.seed(seed)

    kwargs: dict = {}
    if baseline_name == "mean" and train_csv:
        kwargs["train_csv"] = train_csv
    if baseline_name == "geo_prior":
        kwargs["seed"] = seed

    predictor = BASELINES[baseline_name](**kwargs)

    results = []
    start = time.time()
    for _, row in df.iterrows():
        pred_lat, pred_lon = predictor.predict()
        results.append({
            "true_lat": row["latitude"],
            "true_lon": row["longitude"],
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
        })
    elapsed = time.time() - start

    metrics = calculate_metrics(results)

    report = {
        "baseline": baseline_name,
        "csv": csv_path,
        "n_samples": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics,
    }

    _print_report(report)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nReport saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    m = report["metrics"]
    print(f"\nBaseline : {report['baseline'].upper()}")
    print("-" * 42)
    for t in [1, 25, 200, 750, 2500]:
        key = f"acc_@{t}km"
        if key in m:
            print(f"  {key:<22}: {m[key] * 100:.1f}%")
    print()
    for key in ["median_error_km", "mean_error_km"]:
        if key in m:
            print(f"  {key:<22}: {m[key]:.0f} km")
    print(f"\n  Samples  : {report['n_samples']}")
    print(f"  Elapsed  : {report['elapsed_seconds']}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a baseline predictor on a benchmark CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nBaselines\n"
            "---------\n"
            "  random    Uniformly random lat/lon  (≈5000 km median error)\n"
            "  mean      Always predict training-set centroid  (≈4000 km)\n"
            "  geo_prior Random point on land  (≈3500 km)\n"
        ),
    )
    parser.add_argument(
        "--baseline", choices=sorted(BASELINES.keys()), required=True,
        help="Baseline predictor to evaluate",
    )
    parser.add_argument(
        "--csv", required=True,
        help="Benchmark CSV path (columns: latitude, longitude)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the JSON report",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--train_csv", default=None,
        help="Training CSV for the 'mean' baseline (to compute mean coords)",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_baseline(
        args.baseline,
        args.csv,
        args.output,
        seed=args.seed,
        train_csv=args.train_csv,
    )
