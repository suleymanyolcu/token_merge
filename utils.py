import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def has_usable_logits(model: nn.Module) -> bool:
    return bool(getattr(model, "num_classes", 0) > 0 and not isinstance(getattr(model, "head", None), nn.Identity))


def _get_interpolation(name: str) -> InterpolationMode:
    mapping = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }
    return mapping.get(name, InterpolationMode.BICUBIC)


def build_eval_loader(
    model: nn.Module,
    data_dir: str,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
):
    data_config = resolve_model_data_config(model)
    image_size = data_config["input_size"][1:]
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=_get_interpolation(data_config.get("interpolation", "bicubic"))),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ]
    )

    dataset = datasets.CIFAR100(root=data_dir, train=False, transform=transform, download=True)
    if num_samples and num_samples < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, data_config


def extract_features_and_logits(model: nn.Module, images: torch.Tensor):
    tokens = model.forward_features(images)
    if hasattr(model, "forward_head"):
        pooled = model.forward_head(tokens, pre_logits=True)
        logits = model.forward_head(tokens, pre_logits=False) if has_usable_logits(model) else None
    else:
        pooled = model(images)
        logits = None

    if pooled.ndim > 2:
        pooled = pooled.flatten(1)
    return pooled, logits


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_benchmark(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    baseline_features: torch.Tensor = None,
    baseline_preds: torch.Tensor = None,
):
    model.eval()
    model.to(device)

    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= warmup_batches:
            break
        images = images.to(device, non_blocking=device.type == "cuda")
        extract_features_and_logits(model, images)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cached_features = []
    cached_preds = []
    cosine_sum = 0.0
    cosine_count = 0
    agreement_sum = 0
    agreement_count = 0
    batch_times = []
    sample_cosines = []
    sample_pred_agree = []
    feature_offset = 0
    total_images = 0

    _sync(device)
    run_start = time.perf_counter()
    for images, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")

        _sync(device)
        batch_start = time.perf_counter()
        features, logits = extract_features_and_logits(model, images)
        _sync(device)
        batch_times.append(time.perf_counter() - batch_start)

        features_cpu = features.detach().float().cpu()
        batch_size = features_cpu.shape[0]
        total_images += batch_size

        if baseline_features is None:
            cached_features.append(features_cpu)
        else:
            reference = baseline_features[feature_offset: feature_offset + batch_size]
            batch_cosine = F.cosine_similarity(features_cpu, reference, dim=1)
            cosine_sum += batch_cosine.sum().item()
            cosine_count += batch_size
            sample_cosines.extend(batch_cosine.tolist())

        if logits is not None:
            preds = logits.argmax(dim=1).detach().cpu()
            if baseline_preds is None:
                cached_preds.append(preds)
            else:
                reference_preds = baseline_preds[feature_offset: feature_offset + batch_size]
                batch_agree = preds == reference_preds
                agreement_sum += batch_agree.sum().item()
                agreement_count += batch_size
                sample_pred_agree.extend(batch_agree.to(dtype=torch.int32).tolist())

        feature_offset += batch_size

    _sync(device)
    total_elapsed = time.perf_counter() - run_start

    peak_memory_mb = float("nan")
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = {
        "total_elapsed_s": total_elapsed,
        "throughput_images_per_s": total_images / total_elapsed if total_elapsed > 0 else float("nan"),
        "mean_latency_ms": (sum(batch_times) / len(batch_times)) * 1000.0 if batch_times else float("nan"),
        "peak_gpu_memory_mb": peak_memory_mb,
        "mean_feature_cosine": cosine_sum / cosine_count if cosine_count else float("nan"),
        "top1_agreement": agreement_sum / agreement_count if agreement_count else float("nan"),
    }
    detail = {
        "batch_latency_ms": [value * 1000.0 for value in batch_times],
        "sample_cosine": sample_cosines,
        "sample_pred_agree": sample_pred_agree,
    }

    stored_features = torch.cat(cached_features, dim=0) if cached_features else None
    stored_preds = torch.cat(cached_preds, dim=0) if cached_preds, else None
    return result, stored_features, stored_preds, detail


def save_plots(df, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("throughput_images_per_s", "Throughput (images/s)", output_dir / "throughput_vs_r.png"),
        ("mean_latency_ms", "Mean latency per batch (ms)", output_dir / "latency_vs_r.png"),
        ("peak_gpu_memory_mb", "Peak GPU memory (MB)", output_dir / "memory_vs_r.png"),
        ("mean_feature_cosine", "Feature cosine similarity", output_dir / "feature_similarity_vs_r.png"),
    ]

    for column, ylabel, path in plot_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        valid = df[["r", column]].dropna()
        if valid.empty:
            ax.text(0.5, 0.5, "Not available on this run", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.plot(valid["r"], valid[column], marker="o")
            ax.set_xlabel("ToMe r")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
