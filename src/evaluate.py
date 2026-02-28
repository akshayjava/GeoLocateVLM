import argparse
import os
import re
import sys

import pandas as pd
from geopy.distance import great_circle
from tqdm import tqdm

def parse_coordinates(text):
    """
    Extracts latitude and longitude from text.
    Expected format: "City, Country <lat, lon>" or just "lat, lon"
    """
    # Look for patterns like (lat, lon) or <lat, lon> or just lat, lon
    # Matches integers or decimals e.g. "48, 2" or "-33.87, 151.21"
    match = re.search(r"(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
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

def evaluate(csv_path, model_path="models/geolocate_vlm", image_dir="data/images"):
    from src.inference import GeoLocator  # lazy: avoids importing torch at module load
    df = pd.read_csv(csv_path)
    locator = GeoLocator(model_path=model_path)
    
    results = []
    
    print(f"Evaluating on {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{idx}.jpg" # Assuming images are named by index as in data_prep
        # If image paths are in CSV, use that
        if 'image_path' in row:
             image_path = row['image_path']
        
        try:
            pred_text = locator.predict(image_path)
            pred_coords = parse_coordinates(pred_text)
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="models/geolocate_vlm")
    parser.add_argument("--image_dir", type=str, default="data/images")
    args = parser.parse_args()
    
    evaluate(args.csv, args.model_path, args.image_dir)
