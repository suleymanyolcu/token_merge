import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Markdown report from a benchmark output directory.")
    parser.add_argument("--input-dir", required=True, help="Benchmark output directory containing metrics.csv.")
    parser.add_argument("--title", default="ToMe vs Frozen MAE ViT on CIFAR-100", help="Report title.")
    parser.add_argument("--output", default=None, help="Markdown output path. Defaults to <input-dir>/report.md.")
    return parser.parse_args()


def load_detail_frames(detail_dir: Path, prefix: str):
    frames = []
    for path in sorted(detail_dir.glob(f"{prefix}_r*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_histograms(input_dir: Path, metrics: pd.DataFrame):
    detail_dir = input_dir / "details"
    asset_dir = input_dir / "report_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    batch_df = load_detail_frames(detail_dir, "batch_details")
    sample_df = load_detail_frames(detail_dir, "sample_details")

    created = []

    if not batch_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in sorted(batch_df["r"].unique()):
            subset = batch_df.loc[batch_df["r"] == r, "latency_ms"]
            ax.hist(subset, bins=min(20, max(5, len(subset))), alpha=0.45, label=f"r={r}")
        ax.set_xlabel("Batch latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Batch Latency Distribution")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = asset_dir / "batch_latency_histogram.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(path)

    if not sample_df.empty and "cosine_similarity" in sample_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        non_baseline = sample_df[sample_df["r"] != 0]
        target_rs = sorted(non_baseline["r"].unique()) if not non_baseline.empty else sorted(sample_df["r"].unique())
        for r in target_rs:
            subset = sample_df.loc[sample_df["r"] == r, "cosine_similarity"]
            ax.hist(subset, bins=30, alpha=0.4, label=f"r={r}")
        ax.set_xlabel("Cosine similarity to baseline features")
        ax.set_ylabel("Count")
        ax.set_title("Per-sample Feature Similarity Distribution")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = asset_dir / "feature_cosine_histogram.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(path)

    if not sample_df.empty and "pred_agree" in sample_df.columns:
        agree_df = sample_df.groupby("r", as_index=False)["pred_agree"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(agree_df["r"].astype(str), agree_df["pred_agree"])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("ToMe r")
        ax.set_ylabel("Agreement rate")
        ax.set_title("Prediction Agreement by r")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = asset_dir / "prediction_agreement_by_r.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(path)

    if not metrics.empty:
        baseline_tp = metrics.loc[metrics["r"] == 0, "throughput_images_per_s"].iloc[0]
        relative = metrics[["r", "throughput_images_per_s"]].copy()
        relative["speedup_vs_baseline"] = relative["throughput_images_per_s"] / baseline_tp
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(relative["r"].astype(str), relative["speedup_vs_baseline"])
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("ToMe r")
        ax.set_ylabel("Relative throughput")
        ax.set_title("Throughput Relative to Baseline")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = asset_dir / "throughput_relative_to_baseline.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(path)

    return created


def relpath(path: Path, start: Path) -> str:
    return str(path.relative_to(start))


def format_markdown_table(df: pd.DataFrame, float_cols):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                cells.append("NA")
            elif col in float_cols:
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(title: str, input_dir: Path, metrics: pd.DataFrame):
    baseline = metrics.loc[metrics["r"] == 0].iloc[0]
    best_tp = metrics.loc[metrics["throughput_images_per_s"].idxmax()]
    best_lat = metrics.loc[metrics["mean_latency_ms"].idxmin()]
    max_r_row = metrics.loc[metrics["r"].idxmax()]

    throughput_change = (best_tp["throughput_images_per_s"] / baseline["throughput_images_per_s"] - 1.0) * 100.0
    latency_change = (best_lat["mean_latency_ms"] / baseline["mean_latency_ms"] - 1.0) * 100.0

    lines = [
        f"# {title}",
        "",
        "## Scope",
        "This report summarizes an inference-only comparison between a frozen MAE-style ViT baseline and ToMe-patched variants on CIFAR-100 test images.",
        "No training, fine-tuning, linear probing, or hyperparameter search was performed.",
        "CIFAR-100 is used only as an image dataset for benchmarking frozen representations.",
        "",
        "## Experimental Setup",
        f"- Model preset: `{baseline['model_preset']}`",
        f"- Model identifier: `{baseline['model_name']}`",
        f"- Device: `{baseline['device']}`",
        f"- Batch size: `{int(baseline['batch_size'])}`",
        f"- Evaluated samples: `{int(baseline['num_samples'])}`",
        f"- Input resolution: `{int(baseline['input_height'])}x{int(baseline['input_width'])}`",
        f"- Parameter count: `{int(baseline['param_count']):,}`",
        "",
        "## Metrics",
        "- Throughput: images processed per second.",
        "- Mean latency: average wall-clock latency per batch.",
        "- Peak GPU memory: maximum allocated CUDA memory during a run when CUDA is available.",
        "- Feature cosine similarity: similarity between baseline pooled features and ToMe pooled features on the same images.",
        "- Top-1 agreement: baseline-vs-ToMe prediction agreement when the selected checkpoint exposes usable logits.",
        "",
        "## Aggregate Results",
        format_markdown_table(
            metrics[
                [
                    "r",
                    "variant",
                    "throughput_images_per_s",
                    "mean_latency_ms",
                    "peak_gpu_memory_mb",
                    "mean_feature_cosine",
                    "top1_agreement",
                ]
            ],
            {
                "throughput_images_per_s",
                "mean_latency_ms",
                "peak_gpu_memory_mb",
                "mean_feature_cosine",
                "top1_agreement",
            },
        ),
        "",
        "## Analysis",
        f"- Best throughput occurs at `r={int(best_tp['r'])}` with a change of `{throughput_change:+.2f}%` relative to the baseline.",
        f"- Lowest mean latency occurs at `r={int(best_lat['r'])}` with a change of `{latency_change:+.2f}%` relative to the baseline.",
        f"- At the largest tested reduction `r={int(max_r_row['r'])}`, mean feature cosine similarity is `{max_r_row['mean_feature_cosine']:.4f}`.",
    ]

    if pd.notna(max_r_row["peak_gpu_memory_mb"]) and pd.notna(baseline["peak_gpu_memory_mb"]):
        memory_change = max_r_row["peak_gpu_memory_mb"] - baseline["peak_gpu_memory_mb"]
        lines.append(f"- Peak GPU memory changes by `{memory_change:+.2f} MB` between baseline and the largest tested reduction.")
    else:
        lines.append("- Peak GPU memory is not discussed here because the selected run was CPU-only or did not record CUDA memory.")

    if metrics["top1_agreement"].notna().any():
        agree_row = metrics[metrics["top1_agreement"].notna()].iloc[-1]
        lines.append(
            f"- Prediction agreement remains `{agree_row['top1_agreement']:.4f}` at `r={int(agree_row['r'])}`. "
            "This is agreement with the baseline model, not CIFAR-100 accuracy."
        )
    else:
        lines.append("- Prediction agreement is unavailable for this preset because the checkpoint does not expose a usable classification head.")

    lines.extend(
        [
            "",
            "## Figures",
            f"![Throughput vs r]({relpath(input_dir / 'throughput_vs_r.png', input_dir)})",
            "",
            f"![Latency vs r]({relpath(input_dir / 'latency_vs_r.png', input_dir)})",
            "",
            f"![Peak memory vs r]({relpath(input_dir / 'memory_vs_r.png', input_dir)})",
            "",
            f"![Feature similarity vs r]({relpath(input_dir / 'feature_similarity_vs_r.png', input_dir)})",
            "",
        ]
    )

    asset_dir = input_dir / "report_assets"
    extra_assets = [
        asset_dir / "throughput_relative_to_baseline.png",
        asset_dir / "batch_latency_histogram.png",
        asset_dir / "feature_cosine_histogram.png",
        asset_dir / "prediction_agreement_by_r.png",
    ]

    lines.append("## Histograms and Distribution Views")
    for asset in extra_assets:
        if asset.exists():
            caption = asset.stem.replace("_", " ").title()
            lines.append(f"![{caption}]({relpath(asset, input_dir)})")
            lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "- Higher throughput and lower latency indicate more efficient inference.",
            "- Feature similarity close to 1.0 indicates that ToMe preserves the baseline representation more closely.",
            "- If agreement is present, it only measures whether the ToMe-patched model preserves the baseline ImageNet-head prediction on the same CIFAR-100 images.",
            "- This report should be interpreted as a first-pass efficiency study rather than a final accuracy benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    metrics_path = input_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics.csv in {input_dir}")

    detail_dir = input_dir / "details"
    if not detail_dir.exists():
        raise FileNotFoundError(
            f"Could not find {detail_dir}. Re-run the benchmark with the updated evaluation script to save report-ready detail files."
        )

    metrics = pd.read_csv(metrics_path).sort_values("r").reset_index(drop=True)
    save_histograms(input_dir, metrics)

    report_text = build_report(args.title, input_dir, metrics)
    output_path = Path(args.output) if args.output else input_dir / "report.md"
    output_path.write_text(report_text)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
