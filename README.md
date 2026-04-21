# ToMe on Frozen MAE ViT for CIFAR-100

This repository is a small evaluation-only prototype for testing whether **Token Merging (ToMe)** makes a **frozen MAE-based Vision Transformer** more efficient on **CIFAR-100** images.

The project does not do training, fine-tuning, linear probing, or hyperparameter search. CIFAR-100 is used only as an image dataset for benchmarking pretrained models at inference time.

## What this project does

- loads the CIFAR-100 test split from `torchvision`
- resizes inputs to the ViT input size
- runs a baseline MAE-style ViT
- runs the same model with ToMe applied for several `r` values
- measures throughput, latency, feature drift, and optional prediction agreement
- saves CSV metrics, plots, detail files, and a generated Markdown report

## Model options

- `mae_base_backbone`
  - MAE-pretrained `timm` ViT backbone
  - useful for feature comparisons
  - no usable classification head
- `mae_base_finetuned_in1k`
  - ViT-Base/16 with official MAE fine-tuned ImageNet-1K weights
  - exposes logits, so top-1 agreement can be measured

Important: even with the classifier-head preset, this is still **not CIFAR-100 accuracy evaluation**. The classifier head predicts ImageNet-1K classes. The reported `top1_agreement` only measures whether ToMe keeps the same top-1 prediction as the baseline on the same CIFAR-100 images.

## Files

- `eval_tome_mae_cifar100.py`: main benchmark script
- `generate_report.py`: builds a Markdown report with plots and histograms
- `utils.py`: data loading, timing, metrics, and plotting helpers
- `tome_patch.py`: minimal ToMe patch adapted for modern `timm`
- `requirements.txt`: pinned environment

Main generated outputs:

- `outputs/metrics.csv`: one row per `r`
- `outputs/details/`: per-batch and per-sample detail CSVs
- `outputs/*.png`: throughput, latency, memory, and feature-similarity plots
- `outputs/report.md`: generated report

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Backbone-only quick run:

```bash
python eval_tome_mae_cifar100.py --num-samples 256 --batch-size 32
```

Classifier-head run:

```bash
python eval_tome_mae_cifar100.py \
  --model-preset mae_base_finetuned_in1k \
  --num-samples 10000 \
  --batch-size 128 \
  --r-values 0 4 8 12 16 \
  --output-dir outputs_full_report
```

Generate the report:

```bash
python generate_report.py --input-dir outputs_full_report
```

CPU-only example:

```bash
python eval_tome_mae_cifar100.py --device cpu --batch-size 16 --num-workers 0
```

## Metrics

For each `r`, the benchmark records:

- throughput in images/sec
- mean latency per batch
- peak GPU memory in MB when CUDA is available
- mean cosine similarity between baseline and ToMe features
- top-1 prediction agreement when logits are available

Interpretation:

- `r=0` is the untouched baseline
- `r>0` are ToMe-patched runs
- higher throughput and lower latency mean better efficiency
- cosine similarity close to `1.0` means ToMe stays close to the baseline representation
- top-1 agreement close to `1.0` means ToMe keeps the same prediction as the baseline more often

## Example result summary

From the CPU run in `outputs_full_report/` using:

- dataset: CIFAR-100 test split, 10,000 images
- preset: `mae_base_finetuned_in1k`
- input size: `224x224`
- batch size: `128`
- sweep: `r = [0, 4, 8, 12, 16]`

Aggregate results:

| Setting | Throughput (img/s) | Mean Latency (ms/batch) | Feature Cosine | Top-1 Agreement |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`r=0`) | 41.42 | 2890.73 | 1.0000 | 1.0000 |
| ToMe `r=4` | 35.38 | 3406.80 | 0.9812 | 0.8642 |
| ToMe `r=8` | 40.54 | 2950.92 | 0.9467 | 0.7685 |
| ToMe `r=12` | 47.41 | 2498.21 | 0.8949 | 0.6687 |
| ToMe `r=16` | 56.73 | 2060.62 | 0.7834 | 0.5261 |

## Main takeaway

ToMe is working in this prototype. As `r` increases, runtime behavior changes noticeably, which means token merging is actually being applied.

The tradeoff from the example run is:

- small `r` values (`4`, `8`) do not help on CPU and can be slower than baseline
- larger `r` values (`12`, `16`) improve throughput and reduce latency
- stronger merging also increases feature drift and lowers prediction agreement

In this run:

- best speed came from `r=16`
- a more moderate tradeoff looked like `r=12`

Supporting figures from `outputs_full_report/`:

- `throughput_vs_r.png`
- `latency_vs_r.png`
- `feature_similarity_vs_r.png`
- `report_assets/throughput_relative_to_baseline.png`
- `report_assets/batch_latency_histogram.png`
- `report_assets/feature_cosine_histogram.png`
- `report_assets/prediction_agreement_by_r.png`

## Limitations

- inference-only prototype
- no supervised CIFAR-100 accuracy
- single-model, single-process benchmark
- GPU memory is only reported on CUDA runs
- ToMe is adapted from an older archived repo, so this is a research prototype rather than a production benchmark
- the classifier-head comparison is agreement with the baseline, not ground-truth correctness

## Scope reminder

- frozen pretrained models only
- no training
- no fine-tuning
- no linear probe
- no hyperparameter search

