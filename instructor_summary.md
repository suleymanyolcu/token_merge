# ToMe on a Frozen MAE-Based ViT: Summary for Instructor

## What I did

I built and ran a small **evaluation-only** prototype to test whether **Token Merging (ToMe)** can make a **MAE-based Vision Transformer** more efficient at inference time.

The main goal was:

- compare the **baseline MAE-based ViT**
- against the **same model with ToMe applied**
- using the same CIFAR-100 test images
- without doing any training, fine-tuning, linear probing, or hyperparameter search

This means the project is an **inference efficiency study**, not a full accuracy benchmark.

## Dataset

I used the **CIFAR-100 test split** from `torchvision`.

- Number of images evaluated in the final run: **10,000**
- Original image size: **32x32**
- Images were resized to **224x224** because that is the input size expected by the ViT model

## Model

For the final run, I used the preset:

- `mae_base_finetuned_in1k`

This corresponds to:

- a **ViT-Base / 16** architecture
- with **official MAE fine-tuned ImageNet-1K weights**
- about **86.6 million parameters**

This model is useful because it gives both:

- pooled feature vectors for feature-drift comparisons
- a usable classification head, so I could also compare **baseline vs ToMe prediction agreement**

Important note:

- the classifier head is still an **ImageNet-1K head**
- so the “classification result” in this project is **not CIFAR-100 accuracy**
- it is only **top-1 agreement between the baseline model and the ToMe-patched model**

## How I ran it

I ran the benchmark on **CPU** with:

- batch size: **128**
- number of samples: **10,000**
- ToMe sweep: **r = [0, 4, 8, 12, 16]**

The command was effectively:

```bash
python eval_tome_mae_cifar100.py \
  --model-preset mae_base_finetuned_in1k \
  --num-samples 10000 \
  --batch-size 128 \
  --r-values 0 4 8 12 16 \
  --output-dir outputs_full_report
```

Then I generated the report:

```bash
python generate_report.py --input-dir outputs_full_report
```

## Which outputs I inspected

I inspected both:

- [outputs/metrics.csv](/Users/suleymanyolcu/token_merge/outputs/metrics.csv)
- [outputs_full_report/metrics.csv](/Users/suleymanyolcu/token_merge/outputs_full_report/metrics.csv)

The two full runs are very close numerically, which suggests the results are stable enough for a first-pass CPU experiment.

For the final explanation below, I use **`outputs_full_report/`** because it contains the full report assets and detail files:

- [outputs_full_report/report.md](/Users/suleymanyolcu/token_merge/outputs_full_report/report.md)
- [outputs_full_report/summary.txt](/Users/suleymanyolcu/token_merge/outputs_full_report/summary.txt)
- [outputs_full_report/throughput_vs_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/throughput_vs_r.png)
- [outputs_full_report/latency_vs_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/latency_vs_r.png)
- [outputs_full_report/feature_similarity_vs_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/feature_similarity_vs_r.png)
- [outputs_full_report/report_assets/throughput_relative_to_baseline.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/throughput_relative_to_baseline.png)
- [outputs_full_report/report_assets/batch_latency_histogram.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/batch_latency_histogram.png)
- [outputs_full_report/report_assets/feature_cosine_histogram.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/feature_cosine_histogram.png)
- [outputs_full_report/report_assets/prediction_agreement_by_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/prediction_agreement_by_r.png)

## Metrics I measured

For each ToMe reduction value `r`, I measured:

- throughput in images/second
- mean latency per batch
- feature cosine similarity between baseline and ToMe outputs
- top-1 prediction agreement between baseline and ToMe

Because this was a CPU-only run:

- peak GPU memory was **not available**

## Final results

From [outputs_full_report/metrics.csv](/Users/suleymanyolcu/token_merge/outputs_full_report/metrics.csv):

| Setting | Throughput (img/s) | Mean Latency (ms/batch) | Feature Cosine Similarity | Top-1 Agreement |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`r=0`) | 41.42 | 2890.73 | 1.0000 | 1.0000 |
| ToMe `r=4` | 35.38 | 3406.80 | 0.9812 | 0.8642 |
| ToMe `r=8` | 40.54 | 2950.92 | 0.9467 | 0.7685 |
| ToMe `r=12` | 47.41 | 2498.21 | 0.8949 | 0.6687 |
| ToMe `r=16` | 56.73 | 2060.62 | 0.7834 | 0.5261 |

## What these results mean

### 1. ToMe is actually working

Yes, ToMe is clearly active.

If ToMe had failed or behaved like a no-op, I would expect:

- throughput to stay almost unchanged
- latency to stay almost unchanged
- feature cosine similarity to stay extremely close to `1.0`
- prediction agreement to stay very close to `1.0`

That is **not** what happened.

Instead, as `r` increased:

- throughput changed substantially
- latency changed substantially
- feature similarity dropped steadily
- prediction agreement dropped steadily

So the patch is having a real effect on model behavior.

### 2. Small ToMe settings did not help on CPU

At low reduction values:

- `r=4` was **slower** than the baseline
- `r=8` was still slightly worse than the baseline

This suggests that on CPU, the overhead of matching and merging tokens can outweigh the benefit when the reduction is too small.

This is visible in:

- [throughput_vs_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/throughput_vs_r.png)
- [latency_vs_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/latency_vs_r.png)

### 3. Larger token merging improved efficiency

At higher reduction levels, ToMe started helping:

- `r=12` increased throughput from **41.42** to **47.41 img/s**
- `r=16` increased throughput from **41.42** to **56.73 img/s**

Relative to baseline:

- `r=12` gave roughly **+14.5% throughput**
- `r=16` gave roughly **+37.0% throughput**

Latency also improved:

- baseline: **2890.73 ms/batch**
- `r=16`: **2060.62 ms/batch**

That is about a **28.7% latency reduction**.

This is also shown clearly in:

- [throughput_relative_to_baseline.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/throughput_relative_to_baseline.png)

### 4. Efficiency gains came with output drift

The price of stronger token merging is that the model output becomes less similar to the baseline.

Feature cosine similarity changed like this:

- `r=4`: **0.9812**
- `r=8`: **0.9467**
- `r=12`: **0.8949**
- `r=16`: **0.7834**

So at high `r`, the ToMe representation is no longer very close to the original baseline representation.

This trend appears in:

- [feature_similarity_vs_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/feature_similarity_vs_r.png)

and the distribution is shown in:

- [feature_cosine_histogram.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/feature_cosine_histogram.png)

From the detailed per-sample statistics:

- at `r=4`, most cosine values remain high
- at `r=16`, the distribution shifts downward a lot
- for `r=16`, the 10th percentile cosine was about **0.706**
- the minimum cosine observed was about **0.463**

So the drift is not just a small average change; some examples move quite a lot.

### 5. Classifier agreement also dropped as `r` increased

Because I used the MAE fine-tuned ImageNet classifier preset, I could also compare the predicted top-1 class from:

- the baseline model
- the ToMe-patched model

Agreement with baseline:

- `r=4`: **0.8642**
- `r=8`: **0.7685**
- `r=12`: **0.6687**
- `r=16`: **0.5261**

This means:

- with `r=16`, the ToMe-patched model matched the baseline prediction on only about **52.6%** of the 10,000 images

This is shown in:

- [prediction_agreement_by_r.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/prediction_agreement_by_r.png)

Again, this is **not CIFAR-100 accuracy**.
It is only **baseline-vs-ToMe agreement** using an ImageNet classifier head on CIFAR-100 images.

## What the histograms show

### Batch latency histogram

See:

- [batch_latency_histogram.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/batch_latency_histogram.png)

This histogram shows that the whole batch-latency distribution moves downward at higher `r`.

For example:

- baseline mean batch latency: about **2891 ms**
- `r=16` mean batch latency: about **2061 ms**

So the improvement is not coming from only one or two lucky batches. The whole latency distribution improves at high `r`.

### Feature cosine histogram

See:

- [feature_cosine_histogram.png](/Users/suleymanyolcu/token_merge/outputs_full_report/report_assets/feature_cosine_histogram.png)

This histogram shows a gradual shift:

- `r=4` stays tightly concentrated near 1.0
- `r=8` is still fairly close to the baseline
- `r=12` starts to spread lower
- `r=16` shifts much lower and becomes noticeably broader

That matches the expected tradeoff:

- more token merging
- more speed
- less faithful output preservation

## Practical conclusion

My main conclusion is:

- **Token Merging does improve inference efficiency in this setup, but only when the reduction is strong enough**

On CPU:

- low `r` values (`4`, `8`) are not worth it
- higher `r` values (`12`, `16`) improve throughput and reduce latency

However:

- stronger token merging also changes the output more
- feature similarity drops steadily
- classifier agreement with the baseline also drops steadily

So the project shows a clear **efficiency vs fidelity tradeoff**.

## Best interpretation of the run

If I had to summarize the results in one sentence:

> On this CPU-based CIFAR-100 inference experiment with a frozen MAE-based ViT, ToMe only became beneficial at larger reduction values, with `r=16` giving the best speedup but also the largest drop in feature similarity and baseline prediction agreement.

## What I would say is the “best” tradeoff

This depends on what matters more:

- if the priority is **maximum speed**, `r=16` is the best result in this run
- if the priority is **keeping outputs closer to baseline**, `r=12` looks like a more moderate tradeoff

Why `r=12` looks reasonable:

- throughput is already above baseline
- latency is already lower than baseline
- feature similarity is still higher than at `r=16`
- agreement is also higher than at `r=16`

So `r=12` looks like a reasonable middle point, while `r=16` is the aggressive efficiency setting.

## Limitations

- This was a **CPU-only** run
- No GPU memory measurements were available
- CIFAR-100 was only used as an input dataset
- I did **not** evaluate true CIFAR-100 accuracy
- The classifier comparison is only agreement with the baseline, not ground-truth accuracy
- The original ToMe codebase is old, so this is still a lightweight research prototype rather than a production benchmark

## Overall takeaway

The experiment did what it was supposed to do:

- it verified that ToMe can be applied to a MAE-based ViT
- it showed that ToMe changes both efficiency and outputs in a measurable way
- it showed that the effect is not trivial or accidental
- it gave a clear first-pass picture of the tradeoff between:
  - **speed**
  - **latency**
  - **representation preservation**
  - **prediction agreement with the baseline**
