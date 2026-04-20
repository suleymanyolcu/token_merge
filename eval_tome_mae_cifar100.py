import argparse
import copy
from pathlib import Path

import pandas as pd
import timm
import torch

from tome_patch import apply_patch
from utils import (
    build_eval_loader,
    count_parameters,
    has_usable_logits,
    resolve_device,
    run_benchmark,
    save_plots,
    set_seed,
)


MODEL_PRESETS = {
    "mae_base_backbone": {
        "description": "Raw MAE-pretrained timm backbone with pooled features only.",
    },
    "mae_base_finetuned_in1k": {
        "description": "Official MAE fine-tuned ImageNet-1K classifier with usable logits.",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
    },
}

DEFAULT_MODEL_PRESET = "mae_base_backbone"
DEFAULT_R_VALUES = [0, 4, 8, 12, 16]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ToMe on a frozen MAE ViT over CIFAR-100.")
    parser.add_argument(
        "--model-preset",
        default=DEFAULT_MODEL_PRESET,
        choices=sorted(MODEL_PRESETS),
        help="Minimal preset: MAE backbone-only or MAE fine-tuned classifier.",
    )
    parser.add_argument("--data-dir", default="./data", help="Dataset root.")
    parser.add_argument("--output-dir", default="./outputs", help="Where metrics and plots are saved.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of CIFAR-100 test images to evaluate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Warmup batches before timing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible subsets.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--r-values", type=int, nargs="+", default=DEFAULT_R_VALUES, help="ToMe reduction values.")
    return parser.parse_args()


def load_model_from_preset(model_preset: str):
    if model_preset == "mae_base_backbone":
        model = timm.create_model("vit_base_patch16_224.mae", pretrained=True)
        return model, "vit_base_patch16_224.mae"

    if model_preset == "mae_base_finetuned_in1k":
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000, global_pool="avg")
        checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_PRESETS[model_preset]["checkpoint_url"],
            map_location="cpu",
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        return model, "vit_base_patch16_224 + official MAE fine-tuned IN1K weights"

    raise ValueError(f"Unknown model preset: {model_preset}")


def build_summary(df: pd.DataFrame, device: torch.device, model_name: str, param_count: int, image_size, batch_size: int):
    best_tp = df.loc[df["throughput_images_per_s"].idxmax()]
    best_lat = df.loc[df["mean_latency_ms"].idxmin()]

    lines = [
        f"Model: {model_name}",
        f"Device: {device}",
        f"Params: {param_count:,}",
        f"Input resolution: {image_size[0]}x{image_size[1]}",
        f"Batch size: {batch_size}",
        f"Best throughput: r={int(best_tp['r'])} -> {best_tp['throughput_images_per_s']:.2f} images/s",
        f"Lowest latency: r={int(best_lat['r'])} -> {best_lat['mean_latency_ms']:.2f} ms/batch",
    ]

    valid_memory = df["peak_gpu_memory_mb"].dropna()
    if not valid_memory.empty:
        first_mem = valid_memory.iloc[0]
        last_mem = valid_memory.iloc[-1]
        lines.append(f"Peak GPU memory trend: {first_mem:.2f} MB -> {last_mem:.2f} MB")
    else:
        lines.append("Peak GPU memory trend: not available on CPU runs")

    feature_start = df.loc[df["r"] == 0, "mean_feature_cosine"].iloc[0]
    feature_end = df.loc[df["r"] == df["r"].max(), "mean_feature_cosine"].iloc[0]
    lines.append(f"Feature cosine similarity: {feature_start:.4f} at r=0 -> {feature_end:.4f} at r={int(df['r'].max())}")

    agreement = df["top1_agreement"].dropna()
    if agreement.empty:
        lines.append("Top-1 agreement: not computed because this checkpoint does not expose a usable classification head")
    else:
        lines.append(f"Top-1 agreement: {agreement.iloc[-1]:.4f} at r={int(df['r'][agreement.index[-1]])}")

    return "\n".join(lines) + "\n"


def write_detail_files(output_dir: Path, r: int, detail: dict, num_samples: int, has_logits: bool) -> None:
    detail_dir = output_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    batch_df = pd.DataFrame(
        {
            "batch_index": list(range(len(detail["batch_latency_ms"]))),
            "latency_ms": detail["batch_latency_ms"],
            "r": r,
        }
    )
    batch_df.to_csv(detail_dir / f"batch_details_r{r}.csv", index=False)

    sample_data = {
        "sample_index": list(range(num_samples)),
        "r": r,
        "cosine_similarity": detail["sample_cosine"],
    }
    if has_logits:
        sample_data["pred_agree"] = detail["sample_pred_agree"]
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(detail_dir / f"sample_details_r{r}.csv", index=False)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    try:
        baseline_model, resolved_model_name = load_model_from_preset(args.model_preset)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load preset '{args.model_preset}'. "
            "This prototype expects a MAE-based VisionTransformer checkpoint or official MAE fine-tuned weights."
        ) from exc

    loader, data_config = build_eval_loader(
        model=baseline_model,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    param_count = count_parameters(baseline_model)
    image_size = data_config["input_size"][1:]

    patched_model = copy.deepcopy(baseline_model)
    apply_patch(patched_model, prop_attn=False)

    rows = []

    baseline_metrics, baseline_features, baseline_preds, baseline_detail = run_benchmark(
        model=baseline_model,
        loader=loader,
        device=device,
        warmup_batches=args.warmup_batches,
    )
    baseline_metrics.update(
        {
            "variant": "baseline",
            "r": 0,
            "model_name": resolved_model_name,
            "model_preset": args.model_preset,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_samples": len(loader.dataset),
            "input_height": image_size[0],
            "input_width": image_size[1],
            "param_count": param_count,
        }
    )
    baseline_metrics["mean_feature_cosine"] = 1.0
    if has_usable_logits(baseline_model):
        baseline_metrics["top1_agreement"] = 1.0
    rows.append(baseline_metrics)

    usable_logits = has_usable_logits(baseline_model)
    baseline_detail["sample_cosine"] = [1.0] * len(loader.dataset)
    if usable_logits:
        baseline_detail["sample_pred_agree"] = [1] * len(loader.dataset)
    write_detail_files(output_dir, 0, baseline_detail, len(loader.dataset), usable_logits)

    for r in sorted(set(args.r_values)):
        if r == 0:
            continue

        patched_model.r = r
        metrics, _, _, detail = run_benchmark(
            model=patched_model,
            loader=loader,
            device=device,
            warmup_batches=args.warmup_batches,
            baseline_features=baseline_features,
            baseline_preds=baseline_preds if usable_logits else None,
        )
        metrics.update(
            {
                "variant": "tome",
                "r": r,
                "model_name": resolved_model_name,
                "model_preset": args.model_preset,
                "device": str(device),
                "batch_size": args.batch_size,
                "num_samples": len(loader.dataset),
                "input_height": image_size[0],
                "input_width": image_size[1],
                "param_count": param_count,
            }
        )
        rows.append(metrics)
        write_detail_files(output_dir, r, detail, len(loader.dataset), usable_logits)

    df = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    save_plots(df, output_dir)

    summary_text = build_summary(
        df=df,
        device=device,
        model_name=resolved_model_name,
        param_count=param_count,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    (output_dir / "summary.txt").write_text(summary_text)

    display_cols = [
        "r",
        "variant",
        "throughput_images_per_s",
        "mean_latency_ms",
        "peak_gpu_memory_mb",
        "mean_feature_cosine",
        "top1_agreement",
    ]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved metrics to {csv_path}")
    print(f"Saved summary to {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
