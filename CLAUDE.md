# CLAUDE.md — GeoLocateVLM

## Project Overview

**GeoLocateVLM** fine-tunes [PaliGemma 3B](https://huggingface.co/google/paligemma-3b-pt-224), a Vision-Language Model (VLM), to predict the geographic location of an image. Given a photo, the model outputs coordinates or a city/country description.

Key techniques used:
- **QLoRA** (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning
- **4-bit quantization** via BitsAndBytes
- **PEFT LoRA adapters** targeting attention and MLP projection layers

---

## Repository Structure

```
GeoLocateVLM/
├── app.py                       # Gradio web demo (entry point for inference UI)
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Test/dev dependencies
├── benchmarks/
│   └── sample_world.csv         # 150-city benchmark dataset (version-controlled)
├── results/                     # Benchmark JSON reports (version-controlled)
│   ├── README.md                # Results table + reproduction instructions
│   ├── baseline_random.json     # Random baseline results (seed 42)
│   ├── baseline_mean.json       # Mean baseline results (seed 42)
│   └── baseline_geo_prior.json  # Geo-prior baseline results (seed 42)
├── src/
│   ├── data_prep.py             # Download images and build HuggingFace dataset
│   ├── train.py                 # Fine-tune PaliGemma with QLoRA
│   ├── inference.py             # GeoLocator class for predictions
│   ├── evaluate.py              # Metrics: Im2GPS thresholds + per-region breakdown
│   ├── benchmark.py             # Unified benchmark runner
│   ├── benchmark_compare.py     # Multi-run comparison table
│   ├── baselines.py             # Random / mean / geo-prior reference predictors
│   └── benchmarks/
│       ├── im2gps3k.py          # Im2GPS3k dataset loader (Phase 4.3)
│       └── yfcc_val.py          # YFCC-val streaming + offline loader (Phase 4.4)
├── tests/                       # Unit test suite (pytest, ≥75% coverage)
└── notebooks/
    └── train_colab.ipynb        # Google Colab training notebook (no local GPU needed)
```

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training. Inference can run on CPU but will be slow.

### HuggingFace Access

PaliGemma is a gated model. Before training or inference, authenticate:

```bash
huggingface-cli login
```

---

## Workflow

### 1. Prepare Data

```bash
python src/data_prep.py
```

- Downloads sample geo-tagged images
- Creates a CSV with columns: `image_path`, `latitude`, `longitude`, `city`, `country`
- Saves a HuggingFace `Dataset` to `data/processed_dataset/`

### 2. Train

```bash
python src/train.py
```

- Loads `data/processed_dataset/`
- Applies 4-bit quantization + LoRA adapters to PaliGemma 3B
- Trains with HuggingFace `Trainer`
- Saves adapters to `models/geolocate_vlm/`

Key hyperparameters (in `src/train.py`):

| Parameter | Value |
|---|---|
| Base model | `google/paligemma-3b-pt-224` |
| Max steps | 100 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| Quantization | 4-bit NF4 (bfloat16) |

### 3. Run Web Demo

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860`. Upload an image to get a geolocation prediction.

### 4. Evaluate

```bash
python src/evaluate.py --csv data/sample.csv --model_path models/geolocate_vlm
```

Reports accuracy at standard geo-distance thresholds: 1 km, 25 km, 200 km, 750 km, 2500 km, plus median/mean error in km.

### 5. Cloud Training (No Local GPU)

Open `notebooks/train_colab.ipynb` in Google Colab. It streams from the `dalle-mini/YFCC100M_OpenAI_subset` dataset (2000 samples) and trains on a free Colab GPU.

---

## Key Classes & Functions

### `src/inference.py` — `GeoLocator`

```python
from src.inference import GeoLocator

locator = GeoLocator(model_path="models/geolocate_vlm")
result = locator.predict(image)   # PIL.Image → str (location text)
```

- Loads base PaliGemma model + LoRA adapters from `model_path`
- Falls back to zero-shot base model if no adapters are found

### `src/data_prep.py`

- `download_image(url)` — fetches and resizes an image to RGB
- `prepare_dataset(csv_path)` — builds a HuggingFace `Dataset` from a CSV

### `src/evaluate.py`

- Computes great-circle distance between predicted and true coordinates
- Reports hierarchical accuracy thresholds following the Im2GPS benchmark

---

## LoRA Target Modules

The following projection layers are adapted via LoRA:

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Ignored Paths

The following are excluded from version control (see `.gitignore`):

- `.venv/` — virtual environment
- `data/` — downloaded datasets
- `models/` — trained adapters and checkpoints
- `wandb/` — experiment tracking logs
- `__pycache__/`, `*.pyc` — Python bytecode
- `temp_image.jpg` — temporary inference files

---

## Benchmarking

### Running benchmarks

The baselines are deterministic (seed 42) and require only the checked-in
CSV — no GPU, no downloaded images, no trained model.

```bash
# Reproduce all three baselines
python src/baselines.py --baseline random    \
    --csv benchmarks/sample_world.csv --output results/baseline_random.json    --seed 42
python src/baselines.py --baseline mean      \
    --csv benchmarks/sample_world.csv --output results/baseline_mean.json      --seed 42
python src/baselines.py --baseline geo_prior \
    --csv benchmarks/sample_world.csv --output results/baseline_geo_prior.json --seed 42

# View comparison table
python src/benchmark_compare.py results/baseline_*.json

# Full model benchmark (requires trained model + image dataset)
python src/benchmark.py \
    --dataset custom \
    --csv data/benchmarks/im2gps3k.csv \
    --model_path models/geolocate_vlm \
    --output results/im2gps3k_r16.json

# All results together
python src/benchmark_compare.py results/*.json
```

### Current baseline numbers (seed 42, `benchmarks/sample_world.csv`, n=150)

| Run                | @750km | @2500km | median    |
|--------------------|--------|---------|-----------|
| baseline_mean      |  0.7%  |   7.3%  | 7 719 km  |
| baseline_geo_prior |  1.3%  |   9.3%  | 8 696 km  |
| baseline_random    |  2.7%  |   6.7%  | 9 471 km  |

Any trained model must achieve a **lower median error** than `baseline_mean`
(7 719 km) to be considered useful.

### Regression policy — **MANDATORY**

> **If a code change causes any benchmark metric to regress, do NOT merge
> the change. Fix or revert it first.**

Before committing any change to `src/`:

1. Re-run the three baseline commands above.
2. Confirm the JSON results in `results/baseline_*.json` match the committed
   files exactly — baselines are deterministic, so any difference indicates a
   bug in `src/evaluate.py` or `src/baselines.py`.
3. If a trained model is available, also run `src/benchmark.py` and confirm
   `median_error_km` does not increase compared to the last committed result.
4. If a regression is detected:
   - Do **not** proceed with the merge.
   - Diagnose the root cause in the changed code.
   - Fix the regression, re-run benchmarks, confirm improvement, then commit.

---

## Common Issues

**Out of GPU memory during training:**
- Reduce `per_device_train_batch_size` to 1 in `src/train.py`
- Increase `gradient_accumulation_steps` to compensate

**Model not found / 403 from HuggingFace:**
- Run `huggingface-cli login` and accept PaliGemma's license on the model page

**Gradio demo crashes on load:**
- Ensure adapters exist at `models/geolocate_vlm/`; the app expects a trained model

---

## License

MIT
