# ToMe on Frozen MAE ViT for CIFAR-100

This is a minimal evaluation-only prototype for checking whether **Token Merging (ToMe)** makes a **frozen MAE-based ViT backbone** more efficient on **CIFAR-100** inputs.

The experiment intentionally does **not** do training, fine-tuning, linear probing, or hyperparameter search. CIFAR-100 is used only as an image dataset for inference benchmarking on a pretrained backbone.

## What the experiment does

- loads the CIFAR-100 test split
- resizes images to the model input size
- normalizes inputs with the selected pretrained model's config from `timm`
- runs baseline inference with a frozen MAE-based ViT
- runs ToMe-patched inference for a sweep of `r` values
- records:
  - total elapsed inference time
  - throughput in images/sec
  - mean latency per batch
  - peak GPU memory in MB when CUDA is available
  - mean cosine similarity between baseline and ToMe pooled features
  - optional top-1 agreement only when the checkpoint has a usable classification head
- saves CSV metrics, plots, and a short text summary

## Model choice

The default preset is:

```text
mae_base_backbone
```

Available presets:

- `mae_base_backbone`
  - `timm` MAE-pretrained ViT backbone
  - returns pooled features
  - no usable classification head
- `mae_base_finetuned_in1k`
  - modern `timm` ViT-Base architecture
  - loads the official MAE fine-tuned ImageNet-1K checkpoint from the MAE repo
  - exposes usable logits, so top-1 agreement is computed

The second preset does **not** turn this into CIFAR-100 accuracy evaluation. The logits remain ImageNet-1K predictions; the reported agreement is only baseline-vs-ToMe prediction agreement on the same CIFAR-100 images.

You can select the preset with `--model-preset`.

## Files

- `eval_tome_mae_cifar100.py`: main evaluation script
- `generate_report.py`: builds a Markdown report with analysis text, figures, and histograms
- `utils.py`: dataset, timing, plotting, and feature-comparison helpers
- `tome_patch.py`: small vendored subset of ToMe adapted for modern `timm`
- `requirements.txt`: conservative dependency pins
- `outputs/metrics.csv`: one row per `r`
- `outputs/details/`: per-batch and per-sample detail CSVs used for histograms
- `outputs/summary.txt`: short textual summary
- `outputs/throughput_vs_r.png`: throughput plot
- `outputs/latency_vs_r.png`: latency plot
- `outputs/memory_vs_r.png`: memory plot
- `outputs/feature_similarity_vs_r.png`: feature cosine similarity plot
- `outputs/report.md`: generated report after running `generate_report.py`

## Setup

Create a fresh virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Quick smoke test on a small subset:

```bash
python eval_tome_mae_cifar100.py --num-samples 256 --batch-size 32
```

Run the classifier-head preset to also get top-1 agreement:

```bash
python eval_tome_mae_cifar100.py --model-preset mae_base_finetuned_in1k --num-samples 256 --batch-size 32
```

Full CIFAR-100 test split:

```bash
python eval_tome_mae_cifar100.py --num-samples 10000 --batch-size 128
```

CPU-only run:

```bash
python eval_tome_mae_cifar100.py --device cpu --batch-size 16 --num-workers 0
```

Custom ToMe sweep:

```bash
python eval_tome_mae_cifar100.py --r-values 0 4 8 12 16
```

Generate a report from a real benchmark run:

```bash
python generate_report.py --input-dir outputs
```

This creates:

- `outputs/report.md`
- `outputs/report_assets/batch_latency_histogram.png`
- `outputs/report_assets/feature_cosine_histogram.png`
- `outputs/report_assets/throughput_relative_to_baseline.png`
- `outputs/report_assets/prediction_agreement_by_r.png` when logits are available

## Output interpretation

- `metrics.csv`:
  - `r=0` is the untouched baseline model
  - `r>0` rows are the ToMe-patched model
  - `mean_feature_cosine` measures feature drift relative to the baseline on the same images
  - `top1_agreement` is only meaningful when the selected preset exposes logits
- `details/`:
  - `batch_details_r*.csv` stores per-batch latency values
  - `sample_details_r*.csv` stores per-sample cosine similarity and optional prediction agreement
- `summary.txt`:
  - best throughput
  - lowest latency
  - memory trend across the sweep
  - feature-similarity drop as `r` increases
- `report.md`:
  - experiment explanation
  - aggregate table
  - generated analysis text
  - references to line plots and histograms
- plots:
  - simple first-pass visualizations for efficiency vs `r`

## Known limitations

- This is an **inference-efficiency** prototype only. No supervised CIFAR-100 accuracy is measured.
- CIFAR-100 images are only used as inputs to a frozen pretrained backbone.
- The default backbone-only preset does not have a usable classification head, so top-1 agreement is unavailable there.
- The `mae_base_finetuned_in1k` preset downloads the official MAE fine-tuned checkpoint from Meta's public checkpoint host on first use.
- The original ToMe repository is archived and targeted an older `timm` stack. `tome_patch.py` vendors only the minimal merge logic and a small patch adapted to `timm==0.9.16`.
- The patched attention path disables fused attention for the ToMe model because merge metrics need explicit attention tensors. This means the comparison is best interpreted as a simple research probe, not a production-grade microbenchmark.
- GPU memory is only reported when running on CUDA.
- The report generator should be run on a real benchmark directory, not a smoke-test directory. If you do quick validation runs first, write them to a separate output folder and generate the report only from the final run you want to analyze.

## Scope reminder

- frozen pretrained models only
- no training
- no fine-tuning
- no linear probe
- no hyperparameter search

Feature similarity is used here as a lightweight proxy for output preservation while testing whether ToMe improves inference efficiency on CIFAR-100-shaped workloads.
