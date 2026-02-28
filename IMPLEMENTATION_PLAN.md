# GeoLocateVLM — Implementation Plan

## Executive Summary

The codebase has a working skeleton but contains critical correctness bugs, zero test coverage,
no benchmarking infrastructure, and reliability issues that prevent it from being production-ready
or scientifically reproducible. This plan addresses those problems in four sequential phases,
ordered so each phase delivers standalone value while building on the previous one.

**Priority axes (in order):**
1. Correctness — fix bugs that silently corrupt training and evaluation
2. Reliability — make every component robust to bad inputs and failures
3. Testing — enforce correctness through automated tests
4. Measurement — benchmark against public datasets and track improvements over time

---

## Audit of Critical Bugs (must fix before any other work)

| # | File | Bug | Impact |
|---|------|-----|--------|
| B1 | `src/train.py` | Loss computed on full `prompt+answer` sequence; no label masking | Model learns to reproduce the prompt; training is fundamentally broken |
| B2 | `src/train.py` | `fp16=True` in `TrainingArguments` but model loaded in `bfloat16` | Dtype mismatch causes NaN loss on many GPUs |
| B3 | `src/inference.py` | `PaliGemmaProcessor` loaded from `model_path` even when falling back to base model | `FileNotFoundError` on first run before training |
| B4 | `src/inference.py` | Raw generation output returned without stripping the prompt prefix | Downstream coordinate parsing always fails |
| B5 | `src/evaluate.py` | Coordinate regex `(-?\d+\.\d+)` rejects integer coordinates and short decimals | Silent parse failure inflates error metrics |
| B6 | `src/evaluate.py` | Image path assumed to be `{image_dir}/{row_index}.jpg` but `data_prep.py` stores full paths | Every evaluation image load fails |
| B7 | `requirements.txt` | `geopy` used in `evaluate.py` but not listed | `ImportError` on fresh install |

---

## Phase 1 — Critical Bug Fixes & Reliability Foundation
**Goal:** Make the existing pipeline run end-to-end without silent failures.
**Scope:** No new features. Fix what is broken. Add input validation.

### 1.1 Fix training label masking (`src/train.py`)

**Problem:** `collate_fn` sets `labels = input_ids`, so the model optimises loss on both
the question tokens and the answer tokens. The model wastes capacity memorising the prompt.

**Fix:**
```python
def collate_fn(examples):
    images  = [ex["image"].convert("RGB") for ex in examples]
    prompts = [ex["prompt"] for ex in examples]
    targets = [ex["target"] for ex in examples]

    # Tokenise prompts only to find the cut-point
    prompt_enc = processor(text=prompts, images=images,
                           return_tensors="pt", padding=True)

    # Tokenise full sequences (prompt + answer)
    full_texts = [f"{p} {t}" for p, t in zip(prompts, targets)]
    full_enc   = processor(text=full_texts, images=images,
                           return_tensors="pt", padding=True)

    labels = full_enc["input_ids"].clone()

    # Mask prompt tokens with -100 so they are ignored by cross-entropy
    prompt_lengths = prompt_enc["attention_mask"].sum(dim=1)
    for i, plen in enumerate(prompt_lengths):
        labels[i, :plen] = -100

    full_enc["labels"] = labels
    return full_enc
```

**Also fix:** Change `fp16=True` → `bf16=True` in `TrainingArguments` to match the model dtype.

### 1.2 Fix processor loading in `GeoLocator` (`src/inference.py`)

Always load the processor from `base_model_id`. Load adapters separately from `model_path`.
Add image existence check and strip prompt from output.

```python
# Always resolve processor from the base model
self.processor = PaliGemmaProcessor.from_pretrained(base_model_id)

# Strip prompt from decoded output
def _strip_prompt(self, text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()
```

### 1.3 Fix coordinate extraction (`src/evaluate.py`)

Replace the narrow regex with one that handles integers, short decimals, and leading minus signs:

```python
COORD_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)"   # lat: optional decimal part
    r"\s*[,;]\s*"           # separator: comma or semicolon
    r"(-?\d+(?:\.\d+)?)"   # lon: optional decimal part
)

def parse_coordinates(text: str) -> tuple[float, float] | None:
    m = COORD_RE.search(text)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None          # reject physically impossible values
    return lat, lon
```

### 1.4 Fix image path lookup in evaluation (`src/evaluate.py`)

Prefer the `image_path` column from the CSV (written by `data_prep.py`) and only fall back
to index-based naming if the column is absent. Validate existence before calling `locator.predict`.

### 1.5 Fix `requirements.txt`

- Add `geopy>=2.4`
- Pin major versions for the ML stack to a known-good combination:

```
torch>=2.1,<3
transformers>=4.40,<5
accelerate>=0.28,<1
bitsandbytes>=0.43,<1
peft>=0.10,<1
datasets>=2.18,<3
geopy>=2.4,<3
pillow>=10,<11
gradio>=4,<5
pandas>=2,<3
requests>=2.31,<3
tqdm>=4.66,<5
sentencepiece>=0.1.99
protobuf>=3.20,<5
wandb>=0.16,<1
huggingface_hub>=0.22,<1
```

### 1.6 Add input validation helpers (`src/utils/validation.py`)

```python
def validate_csv_columns(df, required): ...   # raise ValueError if missing
def validate_coordinates(lat, lon): ...        # assert bounds
def validate_image_file(path): ...            # PIL.Image.verify() + existence
```

**Deliverables after Phase 1:**
- `python src/data_prep.py` → `python src/train.py` → `python src/evaluate.py` all complete
  without errors on the sample dataset
- No silent data-corruption bugs remain
- `requirements.txt` installs cleanly on a fresh environment

---

## Phase 2 — Test Infrastructure
**Goal:** Automated test suite that guards against regressions introduced in Phases 3 and 4.
**Scope:** Unit tests, integration smoke tests, CI configuration.

### 2.1 Directory layout

```
tests/
├── conftest.py              # shared fixtures (tiny dataset, mock model)
├── unit/
│   ├── test_data_prep.py
│   ├── test_train_collate.py
│   ├── test_inference.py
│   └── test_evaluate.py
└── integration/
    └── test_pipeline_smoke.py
```

### 2.2 Unit tests — `test_evaluate.py` (highest priority)

```python
# Coordinate parsing
@pytest.mark.parametrize("text,expected", [
    ("48.8584, 2.2945",        (48.8584,  2.2945)),
    ("Paris 48, 2",            (48.0,     2.0)),
    ("-33.87; 151.21",         (-33.87,  151.21)),
    ("no coordinates here",    None),
    ("91.0, 0.0",              None),           # invalid lat
    ("0.0, 181.0",             None),           # invalid lon
])
def test_parse_coordinates(text, expected):
    assert parse_coordinates(text) == expected

# Metric calculation
def test_calculate_metrics_perfect():
    results = [{"true_lat": 48.8, "true_lon": 2.3,
                "pred_lat": 48.8, "pred_lon": 2.3}]
    m = calculate_metrics(results)
    assert m["acc_@1km"] == 1.0
    assert m["median_error_km"] == pytest.approx(0.0, abs=1e-3)

def test_calculate_metrics_failed_predictions():
    results = [{"true_lat": 0, "true_lon": 0,
                "pred_lat": None, "pred_lon": None}]
    m = calculate_metrics(results)
    assert m["acc_@2500km"] == 0.0
```

### 2.3 Unit tests — `test_train_collate.py`

```python
def test_labels_mask_prompt_tokens(mock_processor, sample_batch):
    batch = collate_fn(sample_batch)
    # Every position within the prompt span must be -100
    for i, prompt_len in enumerate(sample_batch["prompt_lengths"]):
        assert (batch["labels"][i, :prompt_len] == -100).all()
    # At least one non-masked target token must exist
    assert (batch["labels"] != -100).any()

def test_dtype_consistency(mock_model, sample_batch):
    batch = collate_fn(sample_batch)
    assert batch["pixel_values"].dtype == torch.bfloat16
```

### 2.4 Unit tests — `test_inference.py`

```python
def test_predict_strips_prompt(mock_locator, sample_image):
    result = mock_locator.predict(sample_image)
    assert not result.startswith("Where was this photo taken?")

def test_predict_missing_image(mock_locator, tmp_path):
    with pytest.raises(FileNotFoundError):
        mock_locator.predict(str(tmp_path / "nonexistent.jpg"))

def test_predict_corrupted_image(mock_locator, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(Exception):
        mock_locator.predict(str(bad))
```

### 2.5 Unit tests — `test_data_prep.py`

```python
def test_missing_required_columns(tmp_path):
    df = pd.DataFrame({"image_url": ["http://example.com/img.jpg"]})
    df.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        prepare_dataset(str(tmp_path / "bad.csv"), str(tmp_path))

def test_download_failure_skipped(tmp_path, requests_mock):
    requests_mock.get("http://bad.url/img.jpg", status_code=404)
    # Should not raise; should produce dataset with 0 entries
    result = prepare_dataset(...)
    assert result is None or len(result) == 0

def test_invalid_coordinates_rejected(tmp_path):
    df = make_csv(lat=999, lon=0)   # invalid latitude
    with pytest.raises(ValueError):
        prepare_dataset(...)
```

### 2.6 Integration smoke test (`test_pipeline_smoke.py`)

Uses a tiny 2-image synthetic dataset and a mocked model to verify the full pipeline runs:

```
data_prep → train (1 step) → inference → evaluate
```

Does not require a GPU or HuggingFace token. Uses `pytest-mock` to patch `from_pretrained`.

### 2.7 Add development dependencies

```
# requirements-dev.txt
pytest>=8
pytest-mock>=3.12
requests-mock>=1.11
pytest-cov>=5
black>=24
ruff>=0.4
```

### 2.8 GitHub Actions CI (`.github/workflows/tests.yml`)

```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/unit -v --cov=src --cov-report=term-missing
```

**Deliverables after Phase 2:**
- `pytest tests/unit` passes with ≥80% line coverage on all `src/` modules
- CI blocks merges if tests fail
- No GPU or network access required to run unit tests

---

## Phase 3 — Accuracy Improvements
**Goal:** Improve the quality of the trained model's geolocation predictions.

### 3.1 Improved prompting strategy (`src/data_prep.py`)

The model is asked to predict city+country but evaluated on coordinates. Add a second
prompt variant that requests explicit coordinate output:

```python
PROMPT_STRATEGIES = {
    "city_country": (
        "Where was this photo taken?",
        lambda r: f"{r['city']}, {r['country']}"
    ),
    "coordinates": (
        "Give the GPS coordinates of this photo as decimal latitude and longitude.",
        lambda r: f"{r['latitude']:.4f}, {r['longitude']:.4f}"
    ),
    "combined": (
        "Where was this photo taken? Give the city, country, and GPS coordinates.",
        lambda r: f"{r['city']}, {r['country']} ({r['latitude']:.4f}, {r['longitude']:.4f})"
    ),
}
```

Train separate checkpoints for each strategy and pick the best by validation accuracy.

### 3.2 Data augmentation during training (`src/train.py`)

```python
from torchvision import transforms

AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

# Apply in collate_fn only for the training split
if is_training:
    images = [AUGMENT(img) for img in images]
```

### 3.3 LoRA rank sweep (`src/train.py`)

Expose rank as a CLI argument and run experiments at r ∈ {8, 16, 32}.
Higher rank gives more model capacity at the cost of GPU memory.

```python
def train(..., lora_rank: int = 16, lora_alpha: int = 64):
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, ...)
```

### 3.4 Add vision encoder LoRA (`src/train.py`)

The current config targets only the language decoder. Fine-tuning the vision encoder's
projection can significantly improve grounding of visual features to geographic context:

```python
# Add SigLIP vision encoder attention layers
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "vision_model.encoder.layers.*.self_attn.q_proj",  # vision tower
    "vision_model.encoder.layers.*.self_attn.v_proj",
]
```

### 3.5 Geocoding fallback for city-name predictions (`src/inference.py`)

When the model outputs a city name instead of coordinates, use `geopy.geocoders.Nominatim`
to resolve it to coordinates so evaluation can still compute distance error:

```python
from geopy.geocoders import Nominatim
_geocoder = Nominatim(user_agent="geolocate-vlm")

def text_to_coords(text: str) -> tuple[float, float] | None:
    coords = parse_coordinates(text)
    if coords:
        return coords
    try:
        loc = _geocoder.geocode(text, timeout=5)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return None
```

### 3.6 Longer training with cosine LR schedule

```python
TrainingArguments(
    max_steps=500,                    # was 100
    learning_rate=1e-4,               # reduced for stability
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

**Deliverables after Phase 3:**
- `acc_@200km` improves by ≥10 percentage points over Phase 1 baseline on the sample dataset
- Three prompt strategies benchmarked and the best one documented
- Results saved to `results/prompt_strategy_comparison.json`

---

## Phase 4 — Public Benchmark Integration
**Goal:** Measure model performance on established public datasets so results are comparable
with published work.

### 4.1 Benchmark datasets

| Dataset | Size | License | What it tests |
|---------|------|---------|---------------|
| **Im2GPS3k** | 2,997 images | CC / research | Standard geo-localisation benchmark; results at 1/25/200/750/2500 km |
| **YFCC-val26k** | 26,000 images | CC BY | Scale test; Flickr CC images with GPS EXIF metadata |
| **GWS15k** | 15,000 images | Research | Urban-biased street-level benchmark |
| **MP-16** (test split) | ~4,700 images | Research | Same domain as common training split; use only the held-out test set |

All of these are publicly downloadable or accessible via HuggingFace Datasets.

### 4.2 `src/benchmark.py` — unified benchmark runner

```python
"""
Run the model against a named public benchmark and report Im2GPS-style metrics.

Usage:
    python src/benchmark.py \
        --dataset im2gps3k \
        --model_path models/geolocate_vlm \
        --output results/im2gps3k_run1.json
"""

DATASETS = {
    "im2gps3k":  load_im2gps3k,    # returns iterator of (PIL.Image, lat, lon)
    "yfcc_val":  load_yfcc_val,
    "gws15k":    load_gws15k,
}

def run_benchmark(dataset_name, model_path, output_path, max_samples=None):
    locator  = GeoLocator(model_path=model_path)
    loader   = DATASETS[dataset_name]()
    results  = []

    for image, true_lat, true_lon in tqdm(loader):
        pred_text   = locator.predict(image)
        pred_coords = text_to_coords(pred_text)   # with geocoding fallback (Phase 3.5)
        results.append({
            "true_lat": true_lat, "true_lon": true_lon,
            "pred_text": pred_text,
            "pred_lat": pred_coords[0] if pred_coords else None,
            "pred_lon": pred_coords[1] if pred_coords else None,
        })

    metrics = calculate_metrics(results)
    metrics["dataset"] = dataset_name
    metrics["model_path"] = model_path
    metrics["n_samples"] = len(results)
    metrics["parse_failure_rate"] = sum(
        1 for r in results if r["pred_lat"] is None
    ) / len(results)

    Path(output_path).write_text(json.dumps(metrics, indent=2))
    print_metrics_table(metrics)
    return metrics
```

### 4.3 Im2GPS3k dataset loader (`src/benchmarks/im2gps3k.py`)

```python
def load_im2gps3k(data_dir="data/benchmarks/im2gps3k"):
    """
    Download from: https://github.com/lugiavn/revisiting-im2gps
    CSV: im2gps3k_places365.csv  (image_file, lat, lon)
    Images: im2gps3k_images/
    """
    csv_path = Path(data_dir) / "im2gps3k_places365.csv"
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        img_path = Path(data_dir) / "im2gps3k_images" / row["image_file"]
        if img_path.exists():
            yield Image.open(img_path).convert("RGB"), row["lat"], row["lon"]
```

### 4.4 YFCC-val26k loader (`src/benchmarks/yfcc_val.py`)

```python
def load_yfcc_val(split="validation", max_samples=None):
    """
    Streams directly from HuggingFace Hub — no manual download needed.
    Dataset: dalle-mini/YFCC100M_OpenAI_subset
    """
    ds = load_dataset(
        "dalle-mini/YFCC100M_OpenAI_subset",
        split=split,
        streaming=True,
    )
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        coords = _parse_yfcc_coords(sample.get("tags", ""))
        if coords and sample.get("image"):
            yield sample["image"], coords[0], coords[1]
```

### 4.5 Baseline comparisons

Include baselines in every benchmark run so results are interpretable without reading papers:

| Baseline | Description |
|----------|-------------|
| **Random** | Uniformly random lat/lon; expected median error ≈ 5,000 km |
| **Mean prediction** | Always predict the mean GPS of the training set; median ≈ 4,000 km |
| **PaliGemma zero-shot** | Base model without fine-tuning; upper bound for domain transfer |
| **Geo-prior (land-only)** | Random point on land; expected median ≈ 3,500 km |

### 4.6 Results dashboard (`src/benchmark_compare.py`)

Reads multiple `results/*.json` files and prints a comparison table:

```
┌────────────────────────────────────────────────────────────┐
│  Model                │ @1km  │@25km  │@200km │@750km │med  │
├───────────────────────┼───────┼───────┼───────┼───────┼─────┤
│  Random baseline      │  0.0% │  0.1% │  0.8% │  3.8% │5020 │
│  PaliGemma zero-shot  │  1.2% │  4.5% │ 14.3% │ 32.1% │1840 │
│  GeoLocateVLM (r=8)   │  2.1% │  7.8% │ 21.5% │ 43.2% │1120 │
│  GeoLocateVLM (r=16)  │  2.9% │  9.4% │ 24.7% │ 47.6% │ 980 │
└───────────────────────┴───────┴───────┴───────┴───────┴─────┘
```

### 4.7 Per-region breakdown

Add continent-level and climate-zone breakdowns to identify where the model fails:

```python
def calculate_metrics_by_region(results: list[dict]) -> dict:
    regions = assign_regions(results)   # uses reverse geocoding or lat/lon bucketing
    return {region: calculate_metrics(subset) for region, subset in regions.items()}
```

**Deliverables after Phase 4:**
- `python src/benchmark.py --dataset im2gps3k` runs end-to-end and produces a JSON results file
- Results table committed to `results/README.md` for reproducibility
- Zero-shot baseline run documented for comparison

---

## Implementation Schedule

```
Week 1     Phase 1 — Bug fixes & validation
Week 2     Phase 2 — Unit tests & CI
Week 3-4   Phase 3 — Accuracy improvements (can overlap with Phase 2 QA)
Week 5-6   Phase 4 — Benchmarks (Im2GPS3k first, YFCC-val second)
Ongoing    Results tracking & iteration
```

---

## File Change Summary

| File | Phase | Change type |
|------|-------|-------------|
| `src/train.py` | 1, 3 | Fix label masking; add augmentation, LR schedule, LoRA rank param |
| `src/inference.py` | 1, 3 | Fix processor path; strip prompt; add geocoding fallback |
| `src/evaluate.py` | 1, 3 | Fix regex; fix image paths; add geocoding fallback |
| `src/data_prep.py` | 1, 3 | Add column validation; add prompt strategies |
| `src/utils/validation.py` | 1 | New — shared validation helpers |
| `app.py` | 1 | Add error handling; clean temp files |
| `requirements.txt` | 1 | Add geopy; pin versions |
| `requirements-dev.txt` | 2 | New — dev/test dependencies |
| `tests/unit/*.py` | 2 | New — unit test suite |
| `tests/integration/*.py` | 2 | New — smoke tests |
| `.github/workflows/tests.yml` | 2 | New — CI pipeline |
| `src/benchmark.py` | 4 | New — benchmark runner |
| `src/benchmarks/im2gps3k.py` | 4 | New — Im2GPS3k loader |
| `src/benchmarks/yfcc_val.py` | 4 | New — YFCC-val loader |
| `src/benchmark_compare.py` | 4 | New — results comparison table |
| `results/` | 4 | New — benchmark output directory |

---

## Success Criteria

| Metric | Current | Phase 1 target | Phase 4 target |
|--------|---------|----------------|----------------|
| Pipeline runs end-to-end | No (B1–B7) | Yes | Yes |
| Unit test coverage | 0% | ≥80% | ≥80% |
| Im2GPS3k `acc_@200km` | N/A | Measurable | ≥20% |
| Im2GPS3k `acc_@750km` | N/A | Measurable | ≥40% |
| Parse failure rate | ~100% | <5% | <5% |
| CI blocks bad merges | No | Yes | Yes |
