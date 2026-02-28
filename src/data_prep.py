import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from datasets import Dataset, Image as HFImage
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt strategies (Phase 3.1)
# Each entry is (prompt_text, target_fn) where target_fn receives a CSV row.
# ---------------------------------------------------------------------------

PROMPT_STRATEGIES = {
    "city_country": (
        "Where was this photo taken?",
        lambda r: f"{r['city']}, {r['country']}",
    ),
    "coordinates": (
        "Give the GPS coordinates of this photo as decimal latitude and longitude.",
        lambda r: f"{r['latitude']:.4f}, {r['longitude']:.4f}",
    ),
    "combined": (
        "Where was this photo taken? Give the city, country, and GPS coordinates.",
        lambda r: f"{r['city']}, {r['country']} ({r['latitude']:.4f}, {r['longitude']:.4f})",
    ),
}


def download_image(url, save_path):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def prepare_dataset(csv_path, output_dir, image_dir="images", strategy="city_country"):
    """
    Expects a CSV with columns: 'image_url', 'latitude', 'longitude', 'city', 'country'

    Parameters
    ----------
    csv_path   : Path to the input CSV.
    output_dir : Directory where images and the processed dataset are saved.
    image_dir  : Subdirectory (relative to output_dir) for downloaded images.
    strategy   : Prompt strategy key from PROMPT_STRATEGIES.
                 One of 'city_country', 'coordinates', 'combined'.
    """
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(PROMPT_STRATEGIES)}"
        )
    prompt_text, target_fn = PROMPT_STRATEGIES[strategy]

    df = pd.read_csv(csv_path)

    required_cols = {"image_url", "latitude", "longitude", "city", "country"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)

    data = []

    print(f"Downloading images from {csv_path} (strategy='{strategy}')...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_filename = f"{idx}.jpg"
        image_path = os.path.join(output_dir, image_dir, image_filename)

        # Download if not already on disk
        if not os.path.exists(image_path):
            success = download_image(row['image_url'], image_path)
            if not success:
                continue

        city = row['city'] if pd.notna(row['city']) else "Unknown"
        country = row['country'] if pd.notna(row['country']) else "Unknown"

        # Build a mutable row copy so target_fn can reference clean city/country
        row_data = row.to_dict()
        row_data['city'] = city
        row_data['country'] = country

        target_text = target_fn(row_data)
        coordinates_text = f"{row['latitude']}, {row['longitude']}"

        data.append({
            "image": image_path,
            "prompt": prompt_text,
            "target": target_text,
            "coordinates": coordinates_text,
        })

    if not data:
        print("No images were downloaded or processed. Exiting.")
        return None

    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("image", HFImage())

    try:
        dataset.save_to_disk(os.path.join(output_dir, "processed_dataset"))
        print(f"Dataset saved to {os.path.join(output_dir, 'processed_dataset')}")
    except Exception as e:
        print(f"Failed to save dataset: {e}")

    return dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare geolocation dataset")
    parser.add_argument("--csv", default="data/sample.csv")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument(
        "--strategy",
        choices=list(PROMPT_STRATEGIES),
        default="city_country",
        help="Prompt strategy for training targets (default: city_country)",
    )
    args = parser.parse_args()

    # Create a sample CSV if it doesn't exist yet
    if not os.path.exists(args.csv):
        os.makedirs("data", exist_ok=True)
        sample_data = [
            {"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg", "latitude": 48.8584, "longitude": 2.2945, "city": "Paris", "country": "France"},
            {"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg/800px-View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg", "latitude": 40.7488, "longitude": -73.9857, "city": "New York", "country": "USA"},
            {"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Eiffel_Tower_and_the_Trocadero_Fountains_July_2009.jpg/800px-Eiffel_Tower_and_the_Trocadero_Fountains_July_2009.jpg", "latitude": 48.8584, "longitude": 2.2945, "city": "Paris", "country": "France"},
        ]
        pd.DataFrame(sample_data).to_csv(args.csv, index=False)
        print(f"Created {args.csv}")

    prepare_dataset(args.csv, args.output_dir, strategy=args.strategy)
