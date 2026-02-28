# GeoLocateVLM: On-Device Geolocation with VLMs

Fine-tunes [PaliGemma 3B](https://huggingface.co/google/paligemma-3b-pt-224) to
predict the geographic location of an image using **QLoRA** (4-bit quantization +
LoRA adapters), enabling on-device deployment.

## Features

- **Data Preparation** — download and format geo-tagged image datasets
- **QLoRA Fine-tuning** — memory-efficient training with BitsAndBytes + PEFT
- **Evaluation** — Im2GPS-style accuracy metrics at 1/25/200/750/2500 km
- **Baselines** — random, mean, and geo-prior reference predictors
- **Benchmarking** — unified runner + comparison table across multiple runs
- **Per-region metrics** — continent-level accuracy breakdowns
- **Gradio demo** — browser UI for live inference

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training. Inference can run on CPU (slowly).

```bash
huggingface-cli login   # PaliGemma is gated — accept the licence first
```

---

## Quick Start

### 1. Prepare data

```bash
python src/data_prep.py
```

### 2. Train

```bash
python src/train.py
# Options: --lora_rank 16 --max_steps 500 --strategy coordinates
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --csv data/benchmarks/im2gps3k.csv \
    --model_path models/geolocate_vlm
```

### 4. Run web demo

```bash
python app.py   # → http://localhost:7860
```

---

## Benchmark Results

Baselines measured on **`benchmarks/sample_world.csv`** (150 cities,
all continents, seed 42).  A trained model must beat `baseline_mean` to
be worth deploying.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Run                    │  @1km  │ @25km  │ @200km │ @750km │ @2500km │  median │ fail% │
├────────────────────────┼────────┼────────┼────────┼────────┼────────┼─────────┼───────┤
│▶ baseline_mean          │   0.0% │   0.0% │   0.0% │   0.7% │   7.3% │   7719km │    0% │
│  baseline_geo_prior     │   0.0% │   0.0% │   0.0% │   1.3% │   9.3% │   8696km │    0% │
│  baseline_random        │   0.0% │   0.0% │   0.0% │   2.7% │   6.7% │   9471km │    0% │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

▶ = best median error.  `fail%` = fraction of predictions that could not be
parsed to coordinates.

### Expected model performance (after training)

| Run                 | @200km | @750km | median    |
|---------------------|--------|--------|-----------|
| PaliGemma zero-shot |  ~14%  |  ~32%  | ~1 840 km |
| GeoLocateVLM r=8    |  ~22%  |  ~43%  | ~1 120 km |
| GeoLocateVLM r=16   |  ~25%  |  ~48%  |   ~980 km |

Reproduce baselines:

```bash
python src/baselines.py --baseline random    \
    --csv benchmarks/sample_world.csv --output results/baseline_random.json    --seed 42
python src/baselines.py --baseline mean      \
    --csv benchmarks/sample_world.csv --output results/baseline_mean.json      --seed 42
python src/baselines.py --baseline geo_prior \
    --csv benchmarks/sample_world.csv --output results/baseline_geo_prior.json --seed 42
python src/benchmark_compare.py results/baseline_*.json
```

---

## Repository Structure

```
GeoLocateVLM/
├── app.py                       # Gradio web demo
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Test/dev dependencies
├── benchmarks/
│   └── sample_world.csv         # 150-city benchmark dataset (version-controlled)
├── results/                     # Benchmark JSON reports (version-controlled)
│   ├── README.md                # Results table + reproduction instructions
│   ├── baseline_random.json
│   ├── baseline_mean.json
│   └── baseline_geo_prior.json
├── src/
│   ├── data_prep.py             # Download images & build HuggingFace dataset
│   ├── train.py                 # QLoRA fine-tuning
│   ├── inference.py             # GeoLocator class
│   ├── evaluate.py              # Metrics (Im2GPS thresholds + per-region)
│   ├── benchmark.py             # Unified benchmark runner
│   ├── benchmark_compare.py     # Multi-run comparison table
│   ├── baselines.py             # Random / mean / geo-prior baselines
│   └── benchmarks/
│       ├── im2gps3k.py          # Im2GPS3k dataset loader
│       └── yfcc_val.py          # YFCC-val streaming + offline loader
├── tests/                       # Unit test suite (pytest)
└── notebooks/
    └── train_colab.ipynb        # Google Colab training (no local GPU needed)
```

---

## Model

Base model: [PaliGemma 3B](https://huggingface.co/google/paligemma-3b-pt-224)

Key training hyperparameters:

| Parameter              | Value             |
|------------------------|-------------------|
| LoRA rank              | 16 (default)      |
| LoRA alpha             | 64                |
| Max steps              | 500               |
| Learning rate          | 1e-4 (cosine)     |
| Batch size             | 2 + grad accum ×4 |
| Quantization           | 4-bit NF4 (bf16)  |
| LoRA target modules    | q/k/v/o/gate/up/down\_proj |

---

## License

MIT
