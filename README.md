# GeoLocateVLM: On-Device Geolocation with VLMs

This project implements a pipeline to fine-tune Vision-Language Models (VLMs) like PaliGemma for geolocation tasks, optimized for on-device deployment.

## Features
- **Data Preparation**: Scripts to download and format MP-16/YFCC100M datasets.
- **Fine-tuning**: QLoRA training pipeline for PaliGemma 3B.
- **Inference**: Optimized inference script.
- **Demo**: Gradio web UI.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Data
   ```bash
   python src/data_prep.py
   ```
   This will download sample images to `data/images` and create a Hugging Face dataset in `data/processed_dataset`.

### 2. Train
   ```bash
   python src/train.py
   ```
   This will fine-tune the model and save adapters to `models/geolocate_vlm`.

### 3. Run Demo
   ```bash
   python app.py
   ```

## Model
We use [PaliGemma 3B](https://huggingface.co/google/paligemma-3b-pt-224) as the base model.

## License
MIT
