# Benchmark Results

This directory stores JSON benchmark reports produced by `src/benchmark.py`
and `src/baselines.py`.  Each file is a self-contained run record that can
be compared with `src/benchmark_compare.py`.

---

## Current Results — `benchmarks/sample_world.csv` (150 cities, all continents)

Run with `--seed 42`.  These numbers are the **floor** — any trained model
must beat `baseline_mean` (7 719 km median) to be worth deploying.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Run                    │  @1km  │ @25km  │ @200km │ @750km │ @2500km │  median │ fail% │
├────────────────────────┼────────┼────────┼────────┼────────┼────────┼─────────┼───────┤
│▶ baseline_mean          │   0.0% │   0.0% │   0.0% │   0.7% │   7.3% │   7719km │    0% │
│  baseline_geo_prior     │   0.0% │   0.0% │   0.0% │   1.3% │   9.3% │   8696km │    0% │
│  baseline_random        │   0.0% │   0.0% │   0.0% │   2.7% │   6.7% │   9471km │    0% │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

▶ = best median error on this dataset.

### Expected model targets (once trained)

| Run                 | @200km | @750km | median   |
|---------------------|--------|--------|----------|
| baseline_mean       |  0.0%  |  0.7%  | 7 719 km |
| PaliGemma zero-shot |  ~14%  |  ~32%  | ~1 840 km|
| GeoLocateVLM r=8    |  ~22%  |  ~43%  | ~1 120 km|
| GeoLocateVLM r=16   |  ~25%  |  ~48%  |  ~980 km |

---

## Regression Policy

**A pull request may not be merged if it regresses any previously recorded
metric.**  Specifically:

1. Run `python scripts/run_benchmarks.sh` (or the commands below) before
   every commit that touches `src/`.
2. Compare your results against the last committed JSON files in this
   directory.
3. If `median_error_km` increases **or** any `acc_@Xkm` decreases, the
   change must be fixed or reverted before merging.
4. Baseline numbers (random / mean / geo-prior) are deterministic at
   `--seed 42` and should never change unless `src/evaluate.py` or
   `src/baselines.py` is intentionally modified.

---

## Reproducing baselines

```bash
# From the repository root
python src/baselines.py \
    --baseline random    --csv benchmarks/sample_world.csv \
    --output results/baseline_random.json    --seed 42

python src/baselines.py \
    --baseline mean      --csv benchmarks/sample_world.csv \
    --output results/baseline_mean.json      --seed 42

python src/baselines.py \
    --baseline geo_prior --csv benchmarks/sample_world.csv \
    --output results/baseline_geo_prior.json --seed 42

# Compare all
python src/benchmark_compare.py results/baseline_*.json
```

---

## Running the model benchmark

> Requires a trained model at `models/geolocate_vlm/` and image dataset
> at `data/benchmarks/`.  See the main README for training instructions.

```bash
# Fine-tuned model
python src/benchmark.py \
    --dataset custom \
    --csv data/benchmarks/im2gps3k.csv \
    --model_path models/geolocate_vlm \
    --output results/im2gps3k_r16.json

# Zero-shot (base model, no fine-tuning)
python src/benchmark.py \
    --dataset custom \
    --csv data/benchmarks/im2gps3k.csv \
    --model_path google/paligemma-3b-pt-224 \
    --output results/im2gps3k_zeroshot.json

# Full comparison (baselines + model runs)
python src/benchmark_compare.py results/*.json
```

---

## File naming convention

```
results/
├── baseline_random.json        # Random baseline (seed 42)
├── baseline_mean.json          # Mean-coordinates baseline (seed 42)
├── baseline_geo_prior.json     # Geo-prior (land-only random, seed 42)
├── im2gps3k_zeroshot.json      # PaliGemma zero-shot (no fine-tuning)
├── im2gps3k_r8.json            # Fine-tuned, LoRA rank 8
├── im2gps3k_r16.json           # Fine-tuned, LoRA rank 16
└── im2gps3k_r32.json           # Fine-tuned, LoRA rank 32
```

---

## Report JSON schema

### Model benchmark (`src/benchmark.py`)

```json
{
  "dataset": "im2gps3k",
  "csv": "data/benchmarks/im2gps3k.csv",
  "model_path": "models/geolocate_vlm",
  "n_samples": 2997,
  "n_missing_images": 0,
  "elapsed_seconds": 1842.3,
  "metrics": {
    "acc_@1km": 0.029,
    "acc_@25km": 0.094,
    "acc_@200km": 0.247,
    "acc_@750km": 0.476,
    "acc_@2500km": 0.681,
    "median_error_km": 980.4,
    "mean_error_km": 1523.7,
    "parse_failure_rate": 0.02
  }
}
```

### Baseline report (`src/baselines.py`)

```json
{
  "baseline": "random",
  "csv": "benchmarks/sample_world.csv",
  "n_samples": 150,
  "elapsed_seconds": 0.004,
  "metrics": { ... }
}
```

---

## Im2GPS benchmark thresholds

| Threshold | Label     | Phase 4 target |
|-----------|-----------|----------------|
| 1 km      | Street    | —              |
| 25 km     | City      | —              |
| 200 km    | Region    | ≥ 20%          |
| 750 km    | Country   | ≥ 40%          |
| 2 500 km  | Continent | —              |
