import argparse
import os
import re
import sys

import pandas as pd
from geopy.distance import great_circle
from tqdm import tqdm

# Geocoder is optional; we instantiate it lazily to avoid network calls at import.
_geocoder = None

def _get_geocoder():
    global _geocoder
    if _geocoder is None:
        from geopy.geocoders import Nominatim
        _geocoder = Nominatim(user_agent="geolocate-vlm")
    return _geocoder


def parse_coordinates(text):
    """
    Extracts latitude and longitude from text.
    Expected format: "City, Country <lat, lon>" or just "lat, lon"

    Returns (lat, lon) tuple or None if no valid coordinates found.
    Coordinates outside the physical range (lat [-90,90], lon [-180,180])
    are rejected.
    """
    # Look for patterns like (lat, lon) or <lat, lon> or just lat, lon
    # Matches integers or decimals e.g. "48, 2" or "-33.87, 151.21"
    match = re.search(r"(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    lat, lon = float(match.group(1)), float(match.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon

def text_to_coords(text, use_geocoder=True):
    """
    Convert model output text to (lat, lon), with a geocoding fallback (Phase 3.5).

    First tries to extract decimal coordinates directly from the text.
    If that fails and use_geocoder is True, queries Nominatim to resolve
    a place name to coordinates.

    Parameters
    ----------
    text         : Raw string output from the model.
    use_geocoder : When True, fall back to Nominatim for city/country names.

    Returns
    -------
    (lat, lon) tuple of floats, or None if resolution failed.
    """
    coords = parse_coordinates(text)
    if coords:
        return coords
    if not use_geocoder or not text or not text.strip():
        return None
    try:
        loc = _get_geocoder().geocode(text.strip(), timeout=5)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        pass
    return None


def calculate_metrics(results):
    """
    results: list of dicts with 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
    """
    errors = []
    for res in results:
        if res['pred_lat'] is None or res['pred_lon'] is None:
            errors.append(float('inf')) # Failed prediction
            continue
            
        true_point = (res['true_lat'], res['true_lon'])
        pred_point = (res['pred_lat'], res['pred_lon'])
        
        try:
            dist = great_circle(true_point, pred_point).km
            errors.append(dist)
        except ValueError:
            errors.append(float('inf'))

    if not errors:
        return {}

    thresholds = [1, 25, 200, 750, 2500]
    metrics = {}

    finite_errors = [e for e in errors if e != float('inf')]
    n_total = len(errors)
    n_failed = n_total - len(finite_errors)

    for t in thresholds:
        acc = sum(e <= t for e in errors) / n_total
        metrics[f"acc_@{t}km"] = acc

    metrics["median_error_km"] = float(pd.Series(finite_errors).median()) if finite_errors else float('inf')
    metrics["mean_error_km"] = float(pd.Series(finite_errors).mean()) if finite_errors else float('inf')
    metrics["parse_failure_rate"] = n_failed / n_total

    return metrics

def evaluate(csv_path, model_path="models/geolocate_vlm", image_dir="data/images",
             use_geocoder=True):
    from src.inference import GeoLocator  # lazy: avoids importing torch at module load
    df = pd.read_csv(csv_path)
    locator = GeoLocator(model_path=model_path)

    results = []

    print(f"Evaluating on {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{idx}.jpg"  # fallback: index-based name
        if 'image_path' in row:
            image_path = row['image_path']

        try:
            pred_text = locator.predict(image_path)
            # Use geocoding fallback so city-name outputs still resolve to coords
            pred_coords = text_to_coords(pred_text, use_geocoder=use_geocoder)
            
            res = {
                "true_lat": row['latitude'],
                "true_lon": row['longitude'],
                "pred_text": pred_text,
                "pred_lat": pred_coords[0] if pred_coords else None,
                "pred_lon": pred_coords[1] if pred_coords else None
            }
            results.append(res)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                "true_lat": row['latitude'],
                "true_lon": row['longitude'],
                "pred_text": "Error",
                "pred_lat": None,
                "pred_lon": None
            })
            
    metrics = calculate_metrics(results)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics


# ---------------------------------------------------------------------------
# Per-region metrics (Phase 4.7)
# ---------------------------------------------------------------------------

def _assign_region(lat: float, lon: float) -> str:
    """
    Assign a continent label to a (lat, lon) coordinate using approximate
    bounding boxes.  Returns 'Other' for points that fall outside every box
    (e.g. remote ocean areas).
    """
    if 15.0 <= lat <= 72.0 and -168.0 <= lon <= -52.0:
        return "North America"
    if -56.0 <= lat <= 15.0 and -82.0 <= lon <= -34.0:
        return "South America"
    if 35.0 <= lat <= 71.0 and -10.0 <= lon <= 40.0:
        return "Europe"
    if -35.0 <= lat <= 37.0 and -18.0 <= lon <= 52.0:
        return "Africa"
    if -10.0 <= lat <= 77.0 and 25.0 <= lon <= 145.0:
        return "Asia"
    if -44.0 <= lat <= -10.0 and 113.0 <= lon <= 180.0:
        return "Oceania"
    return "Other"


def calculate_metrics_by_region(results: list) -> dict:
    """
    Compute per-continent accuracy metrics (Phase 4.7).

    Assigns each result to a continent using lat/lon bounding boxes and
    returns a mapping from region name to the same metrics dict produced
    by :func:`calculate_metrics`.

    Parameters
    ----------
    results : list of dicts, each with keys:
              'true_lat', 'true_lon', 'pred_lat', 'pred_lon'

    Returns
    -------
    dict mapping region name (str) → metrics dict (same format as
    :func:`calculate_metrics`).  Regions with zero samples are omitted.
    """
    from collections import defaultdict

    regional: dict = defaultdict(list)
    for res in results:
        region = _assign_region(float(res["true_lat"]), float(res["true_lon"]))
        regional[region].append(res)

    return {
        region: calculate_metrics(subset)
        for region, subset in sorted(regional.items())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="models/geolocate_vlm")
    parser.add_argument("--image_dir", type=str, default="data/images")
    args = parser.parse_args()
    
    evaluate(args.csv, args.model_path, args.image_dir)
