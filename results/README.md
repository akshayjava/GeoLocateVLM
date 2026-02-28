# Benchmark Results

This directory stores JSON benchmark reports produced by `src/benchmark.py`
and `src/baselines.py`.  Each file is a self-contained run record that can
be compared with `src/benchmark_compare.py`.

---

## File naming convention

```
results/
├── baseline_random.json        # Random baseline
├── baseline_mean.json          # Mean-coordinates baseline
├── baseline_geo_prior.json     # Geo-prior (land-only random) baseline
├── im2gps3k_zeroshot.json      # PaliGemma zero-shot (no fine-tuning)
├── im2gps3k_r8.json            # Fine-tuned, LoRA rank 8
├── im2gps3k_r16.json           # Fine-tuned, LoRA rank 16
├── im2gps3k_r32.json           # Fine-tuned, LoRA rank 32
└── yfcc4k_r16.json             # Best model on YFCC4k
```

---

## Reproducing baselines

```bash
# Random baseline
python src/baselines.py \
    --baseline random \
    --csv data/benchmarks/im2gps3k.csv \
    --output results/baseline_random.json

# Mean baseline (uses world centroid if no --train_csv)
python src/baselines.py \
    --baseline mean \
    --csv data/benchmarks/im2gps3k.csv \
    --output results/baseline_mean.json

# Geo-prior baseline
python src/baselines.py \
    --baseline geo_prior \
    --csv data/benchmarks/im2gps3k.csv \
    --output results/baseline_geo_prior.json
```

---

## Running the model benchmark

```bash
# Fine-tuned model
python src/benchmark.py \
    --dataset im2gps3k \
    --csv data/benchmarks/im2gps3k.csv \
    --model_path models/geolocate_vlm \
    --output results/im2gps3k_r16.json

# Zero-shot (base model, no adapters)
python src/benchmark.py \
    --dataset im2gps3k \
    --csv data/benchmarks/im2gps3k.csv \
    --model_path google/paligemma-3b-pt-224 \
    --output results/im2gps3k_zeroshot.json
```

---

## Comparing runs

```bash
python src/benchmark_compare.py results/*.json
```

Expected output format:

```
┌────────────────────────────────────────────────────────────────────┐
│  Run                    │ @1km  │@25km  │@200km │@750km │  med    │
├─────────────────────────┼───────┼───────┼───────┼───────┼─────────┤
│  baseline_random        │  0.0% │  0.1% │  0.8% │  3.8% │ 5020 km │
│  baseline_geo_prior     │  0.0% │  0.2% │  1.1% │  5.2% │ 3480 km │
│  baseline_mean          │  0.0% │  0.0% │  0.9% │  4.1% │ 4050 km │
│  im2gps3k_zeroshot      │  1.2% │  4.5% │ 14.3% │ 32.1% │ 1840 km │
│  im2gps3k_r16  ★        │  2.9% │  9.4% │ 24.7% │ 47.6% │  980 km │
└─────────────────────────┴───────┴───────┴───────┴───────┴─────────┘
★ = best median error
```

---

## Report JSON schema

Each `.json` file produced by `src/benchmark.py` has this structure:

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

Baseline reports produced by `src/baselines.py` use `"baseline"` instead
of `"dataset"` and omit `"n_missing_images"` and `"elapsed_seconds"` for
the model.

---

## Im2GPS benchmark thresholds

The standard Im2GPS evaluation protocol reports accuracy at five distance
thresholds.  The accepted targets for a competitive fine-tuned model are:

| Threshold | Street | City  | Region | Country | Continent |
|-----------|--------|-------|--------|---------|-----------|
| 1 km      | ✓      |       |        |         |           |
| 25 km     |        | ✓     |        |         |           |
| 200 km    |        |       | ✓      |         |           |
| 750 km    |        |       |        | ✓       |           |
| 2500 km   |        |       |        |         | ✓         |

Phase 4 target: `acc_@200km ≥ 20%` and `acc_@750km ≥ 40%`.
