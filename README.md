# ToMe on Frozen MAE ViT for CIFAR-100

This repository is a small evaluation-only prototype for testing whether
**Token Merging (ToMe)** makes a **frozen MAE-based Vision Transformer** more
efficient on **CIFAR-100** images.

The project does not do training, fine-tuning, linear probing, or
hyperparameter search. CIFAR-100 is used only as an image dataset for
benchmarking pretrained models at inference time.

## What this project does

- loads the CIFAR-100 test split from `torchvision`
- resizes inputs from `32x32` to the ViT input size, usually `224x224`
- runs a baseline MAE-style ViT
- runs the same model with ToMe applied for several `r` values
- measures throughput, latency, feature drift, and optional prediction agreement
- saves CSV metrics, detail files, plots, and a text summary

This is an **inference efficiency study**, not a CIFAR-100 accuracy benchmark.

## Model options

- `mae_base_backbone`
  - MAE-pretrained `timm` ViT backbone
  - useful for feature comparisons
  - no usable classification head
- `mae_base_finetuned_in1k`
  - ViT-Base/16 with official MAE fine-tuned ImageNet-1K weights
  - about 86.6 million parameters
  - exposes logits, so top-1 agreement can be measured

Important: even with the classifier-head preset, this is still **not CIFAR-100
accuracy evaluation**. The classifier head predicts ImageNet-1K classes. The
reported `top1_agreement` only measures whether ToMe keeps the same top-1
prediction as the baseline on the same CIFAR-100 images.

## Files

- `eval_tome_mae_cifar100.py`: main benchmark script
- `utils.py`: data loading, timing, metrics, and plotting helpers
- `tome_patch.py`: minimal ToMe patch adapted for modern `timm`
- `requirements.txt`: pinned environment

Main generated outputs:

- `outputs/metrics.csv`: one row per `r`
- `outputs/summary.txt`: short benchmark summary
- `outputs/details/`: per-batch and per-sample detail CSVs
- `outputs/*.png`: throughput, latency, memory, and feature-similarity plots

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
- model: ViT-Base/16 with official MAE fine-tuned ImageNet-1K weights
- parameters: about 86.6 million
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

The earlier `outputs/metrics.csv` run and the full `outputs_full_report/metrics.csv`
run are numerically close, which suggests the result is stable enough for a
first-pass CPU experiment.

## What the results mean

ToMe is clearly active. If it had failed or behaved like a no-op, throughput,
latency, feature cosine similarity, and prediction agreement would remain almost
unchanged. Instead, all of them change as `r` increases.

Small ToMe settings did not help on CPU:

- `r=4` was slower than the baseline
- `r=8` was still slightly worse than the baseline
- the matching and merging overhead can outweigh the benefit when reduction is too small

Larger token merging improved efficiency:

- `r=12` increased throughput from 41.42 to 47.41 img/s, about +14.5%
- `r=16` increased throughput from 41.42 to 56.73 img/s, about +37.0%
- latency dropped from 2890.73 ms/batch at baseline to 2060.62 ms/batch at `r=16`,
  about a 28.7% reduction

The efficiency gains came with output drift:

- feature cosine dropped from 0.9812 at `r=4` to 0.7834 at `r=16`
- top-1 agreement with the baseline dropped from 0.8642 at `r=4` to 0.5261 at `r=16`
- for `r=16`, the 10th percentile cosine was about 0.706 and the minimum observed
  cosine was about 0.463

The histograms in `outputs_full_report/report_assets/` show the same trend:

- batch latency shifts lower at higher `r`
- feature cosine stays close to baseline at `r=4`
- feature cosine spreads lower at `r=12`
- feature cosine shifts much lower and broadens at `r=16`

## Main takeaway

Token Merging improves inference efficiency in this setup only when the reduction
is strong enough. On this CPU-based CIFAR-100 inference experiment with a frozen
MAE-based ViT, `r=16` gave the best speedup but also the largest drop in feature
similarity and baseline prediction agreement.

The practical tradeoff from this run is:

- if maximum speed is the priority, `r=16` is the best result
- if keeping outputs closer to the baseline matters more, `r=12` is a more moderate setting

## Limitations

- inference-only prototype
- CPU-only example run; no GPU memory measurements were available
- no supervised CIFAR-100 accuracy
- CIFAR-100 is only used as an input dataset
- single-model, single-process benchmark
- ToMe is adapted from an older archived repo, so this is a research prototype rather than a production benchmark
- the classifier-head comparison is agreement with the baseline, not ground-truth correctness

## Scope reminder

- frozen pretrained models only
- no training
- no fine-tuning
- no linear probe
- no hyperparameter search
