"""
Benchmark GeoLocateVLM against standard public geolocation datasets.

Supported datasets
------------------
im2gps3k  : 3,000-image Im2GPS test set (Vo et al., 2017).
             Paper: https://arxiv.org/abs/1705.01061
             Download: https://graphics.cs.cmu.edu/projects/im2gps/
             CSV columns required: image_path, latitude, longitude

yfcc4k    : 4,536-image YFCC100M geo-localisation subset
             (Müller-Budack et al., 2018).
             Download: https://github.com/TIBHannover/GeoEstimation
             CSV columns required: image_path, latitude, longitude

custom    : Any CSV with columns: image_path, latitude, longitude

Usage
-----
    python src/benchmark.py \\
        --dataset im2gps3k \\
        --csv     data/benchmarks/im2gps3k.csv \\
        --model_path models/geolocate_vlm \\
        --output  results/im2gps3k_results.json

Preparing Im2GPS3k
------------------
1. Request the dataset from: https://graphics.cs.cmu.edu/projects/im2gps/
2. Download the test images and the accompanying GPS metadata file.
3. Build a CSV with columns:  image_path, latitude, longitude
   (one row per image).
4. Pass that CSV to --csv.

Preparing YFCC4k
----------------
1. Clone the GeoEstimation repository:
       git clone https://github.com/TIBHannover/GeoEstimation
2. Follow its dataset download instructions to obtain the 4k test images.
3. Convert the metadata to a CSV with: image_path, latitude, longitude
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluate import calculate_metrics, parse_coordinates, text_to_coords

SUPPORTED_DATASETS = ["im2gps3k", "yfcc4k", "custom"]

_DATASET_NOTES = {
    "im2gps3k": (
        "Im2GPS3k (Vo et al., 2017) — 3,000 geotagged Flickr images.\n"
        "  Paper   : https://arxiv.org/abs/1705.01061\n"
        "  Download: https://graphics.cs.cmu.edu/projects/im2gps/"
    ),
    "yfcc4k": (
        "YFCC100M 4k subset (Müller-Budack et al., 2018) — 4,536 images.\n"
        "  Download: https://github.com/TIBHannover/GeoEstimation"
    ),
    "custom": "Custom dataset — CSV must have: image_path, latitude, longitude",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_csv(df: pd.DataFrame, dataset: str) -> None:
    """Raise ValueError if the CSV is missing required columns or has out-of-range coordinates."""
    required = {"image_path", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV for '{dataset}' is missing required columns: {missing}. "
            f"Required: {required}"
        )

    bad_lat = df[(df["latitude"] < -90) | (df["latitude"] > 90)]
    if len(bad_lat):
        raise ValueError(
            f"{len(bad_lat)} row(s) have latitude outside [-90, 90]. "
            "Check the CSV for malformed coordinates."
        )

    bad_lon = df[(df["longitude"] < -180) | (df["longitude"] > 180)]
    if len(bad_lon):
        raise ValueError(
            f"{len(bad_lon)} row(s) have longitude outside [-180, 180]. "
            "Check the CSV for malformed coordinates."
        )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_benchmark(
    csv_path: str,
    model_path: str = "models/geolocate_vlm",
    dataset: str = "custom",
    output_path: str | None = None,
    use_geocoder: bool = True,
) -> dict:
    """
    Run the full benchmark and return a report dict.

    Parameters
    ----------
    csv_path    : Path to the benchmark CSV.
    model_path  : Path to fine-tuned LoRA adapters (or HF repo id for zero-shot).
    dataset     : One of SUPPORTED_DATASETS — used only for labelling the report.
    output_path : If given, write the JSON report to this path.

    Returns
    -------
    dict with keys: dataset, csv, model_path, n_samples, n_missing_images,
                    elapsed_seconds, metrics
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark : {dataset.upper()}")
    if dataset in _DATASET_NOTES:
        for line in _DATASET_NOTES[dataset].splitlines():
            print(f"  {line}")
    print(f"{'='*60}\n")

    from src.inference import GeoLocator  # lazy: avoids importing torch at module load

    df = pd.read_csv(csv_path)
    validate_csv(df, dataset)
    print(f"Loaded {len(df)} sample(s) from {csv_path}")

    locator = GeoLocator(model_path=model_path)

    results = []
    missing_images = []
    start = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        image_path = row["image_path"]

        if not os.path.exists(image_path):
            missing_images.append(image_path)
            results.append(
                _failed_result(row["latitude"], row["longitude"], "FILE_NOT_FOUND")
            )
            continue

        try:
            pred_text = locator.predict(image_path)
            # Use geocoding fallback so city-name outputs still resolve to coords
            pred_coords = text_to_coords(pred_text, use_geocoder=use_geocoder)
            results.append({
                "true_lat": row["latitude"],
                "true_lon": row["longitude"],
                "pred_text": pred_text,
                "pred_lat": pred_coords[0] if pred_coords else None,
                "pred_lon": pred_coords[1] if pred_coords else None,
            })
        except Exception as exc:
            results.append(_failed_result(row["latitude"], row["longitude"], f"ERROR: {exc}"))

    elapsed = time.time() - start
    metrics = calculate_metrics(results)

    report = {
        "dataset": dataset,
        "csv": csv_path,
        "model_path": model_path,
        "n_samples": len(df),
        "n_missing_images": len(missing_images),
        "elapsed_seconds": round(elapsed, 1),
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

def _failed_result(true_lat, true_lon, reason):
    return {
        "true_lat": true_lat,
        "true_lon": true_lon,
        "pred_text": reason,
        "pred_lat": None,
        "pred_lon": None,
    }


def _print_report(report):
    m = report["metrics"]
    print("\nBenchmark Results")
    print("-" * 45)
    for threshold in [1, 25, 200, 750, 2500]:
        key = f"acc_@{threshold}km"
        if key in m:
            print(f"  {key:<22}: {m[key]:.4f}  ({m[key]*100:.1f}%)")
    print()
    for key in ["median_error_km", "mean_error_km", "parse_failure_rate"]:
        if key in m:
            print(f"  {key:<22}: {m[key]:.4f}")
    print()
    print(f"  Samples evaluated : {report['n_samples']}")
    print(f"  Missing images    : {report['n_missing_images']}")
    print(f"  Elapsed           : {report['elapsed_seconds']}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    epilog_lines = ["\nDataset notes:"]
    for name, note in _DATASET_NOTES.items():
        epilog_lines.append(f"\n  {name}:\n    " + note.replace("\n", "\n    "))

    parser = argparse.ArgumentParser(
        description="Benchmark GeoLocateVLM on standard public geolocation datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(epilog_lines),
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="custom",
        help="Dataset identifier used to label the report (default: custom)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the benchmark CSV (columns: image_path, latitude, longitude)",
    )
    parser.add_argument(
        "--model_path",
        default="models/geolocate_vlm",
        help="Path to fine-tuned LoRA adapters (default: models/geolocate_vlm)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the JSON report",
    )
    parser.add_argument(
        "--no_geocoder",
        action="store_true",
        help="Disable Nominatim geocoding fallback for city-name predictions",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_benchmark(
        args.csv,
        args.model_path,
        args.dataset,
        args.output,
        use_geocoder=not args.no_geocoder,
    )
