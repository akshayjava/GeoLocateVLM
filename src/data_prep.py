import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from datasets import Dataset, Features, Image as HFImage, Value
from tqdm import tqdm

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

def prepare_dataset(csv_path, output_dir, image_dir="images"):
    """
    Expects a CSV with columns: 'image_url', 'latitude', 'longitude', 'city', 'country'
    """
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)
    
    data = []
    
    print(f"Downloading images from {csv_path}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_filename = f"{idx}.jpg"
        image_path = os.path.join(output_dir, image_dir, image_filename)
        
        # Download if not exists
        if not os.path.exists(image_path):
            success = download_image(row['image_url'], image_path)
            if not success:
                continue
        
        # Create prompt and target
        # We can try different prompt strategies.
        # Strategy 1: "Where was this photo taken?" -> "City, Country"
        # Strategy 2: "Estimate the coordinates." -> "lat, lon"
        
        location_text = f"{row['city']}, {row['country']}"
        coordinates_text = f"{row['latitude']}, {row['longitude']}"
        
        data.append({
            "image": image_path,
            "prompt": "Where was this photo taken?",
            "target": location_text,
            "coordinates": coordinates_text
        })
        
        
    if not data:
        print("No images were downloaded or processed. Exiting.")
        return None

    # Create HF Dataset
    # We store the path to the image, but when loading for training we will load the actual image
    dataset = Dataset.from_list(data)
    
    # Cast image column to Image feature
    dataset = dataset.cast_column("image", HFImage())
    
    try:
        dataset.save_to_disk(os.path.join(output_dir, "processed_dataset"))
        print(f"Dataset saved to {os.path.join(output_dir, 'processed_dataset')}")
    except Exception as e:
        print(f"Failed to save dataset: {e}")
        
    return dataset

if __name__ == "__main__":
    # Example usage
    # Create a dummy CSV if not exists
    if not os.path.exists("data/sample.csv"):
        os.makedirs("data", exist_ok=True)
        sample_data = [
            {"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg", "latitude": 48.8584, "longitude": 2.2945, "city": "Paris", "country": "France"},
            {"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg/800px-View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg", "latitude": 40.7488, "longitude": -73.9857, "city": "New York", "country": "USA"},
             {"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Eiffel_Tower_and_the_Trocadero_Fountains_July_2009.jpg/800px-Eiffel_Tower_and_the_Trocadero_Fountains_July_2009.jpg", "latitude": 48.8584, "longitude": 2.2945, "city": "Paris", "country": "France"},
        ]
        pd.DataFrame(sample_data).to_csv("data/sample.csv", index=False)
        print("Created data/sample.csv")

    prepare_dataset("data/sample.csv", "data")
