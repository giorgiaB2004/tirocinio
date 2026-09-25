from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from data_utils import (
    IGNORE_INDEX,
    build_four_channel_image,
    color_mask_to_index,
    index_to_bgr_mask,
    load_split,
    metrics_from_confusion,
    read_bgr,
    resize_image_and_mask,
    update_confusion_matrix,
)
from model import UNet, build_unet


DEFAULT_COLORMAP_BGR = [
    [0, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
]
DEFAULT_CLASS_NAMES = ["background", "mais", "weeds"]
DEFAULT_CLASS_WEIGHTS = [1.0, 2.21, 5.42]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation checkpoint on the test split"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict_palette",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_joint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument(
        "--fourth_channel",
        type=str,
        default=None,
        choices=["exg", "green_chromaticity"],
    )
    parser.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        help="Load only checkpoint tensors with matching names and shapes.",
    )

    # Fallback architecture options for checkpoints created by older scripts.
    parser.add_argument(
        "--attention_type",
        type=str,
        default="none",
        choices=["none", "class_aware", "dual", "both"],
    )
    parser.add_argument("--use_aspp", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--aspp_branch_channels", type=int, default=256)
    parser.add_argument(
        "--aspp_rates",
        type=int,
        nargs=3,
        default=[2, 4, 6],
    )
    parser.add_argument(
        "--aspp_separable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min_attention", type=float, default=0.25)
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=3,
        default=DEFAULT_CLASS_WEIGHTS,
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")
    return loaded


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=True,
        )
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[UNet, dict[str, Any]]:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        model_config = {
            "num_classes": len(DEFAULT_COLORMAP_BGR),
            "in_channels": 4,
            "attention_type": args.attention_type,
            "class_weights": args.class_weights,
            "dropout_rate": args.dropout_rate,
            "use_aspp": args.use_aspp,
            "base_channels": args.base_channels,
            "aspp_branch_channels": args.aspp_branch_channels,
            "aspp_rates": args.aspp_rates,
            "aspp_separable": args.aspp_separable,
            "min_attention": args.min_attention,
        }

    model = build_unet(**model_config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a valid model state dict")

    if args.allow_partial_checkpoint:
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state
            and hasattr(value, "shape")
            and current_state[key].shape == value.shape
        }
        skipped = sorted(set(state_dict) - set(compatible_state))
        incompatible = model.load_state_dict(compatible_state, strict=False)
        print(f"Partially loaded {len(compatible_state)} tensors.")
        if skipped:
            print(f"Skipped {len(skipped)} incompatible checkpoint tensors.")
        if incompatible.missing_keys:
            print(f"Missing model tensors: {len(incompatible.missing_keys)}")
    else:
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, config


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write: {path}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metric_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean())


def format_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.6f}"


def save_metrics(
    save_dir: Path,
    confusion: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    class_names: Sequence[str],
    mean_latency_seconds: float,
) -> None:
    metric_path = save_dir / "metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["class", "precision", "recall", "f1", "iou", "support"]
        )
        for index, class_name in enumerate(class_names):
            writer.writerow(
                [
                    class_name,
                    format_metric(metric_value(metrics["precision"][index])),
                    format_metric(metric_value(metrics["recall"][index])),
                    format_metric(metric_value(metrics["f1"][index])),
                    format_metric(metric_value(metrics["iou"][index])),
                    int(metric_value(metrics["support"][index])),
                ]
            )

        foreground_slice = slice(1, None)
        writer.writerow(
            [
                "mean_foreground",
                format_metric(finite_mean(metrics["precision"][foreground_slice])),
                format_metric(finite_mean(metrics["recall"][foreground_slice])),
                format_metric(finite_mean(metrics["f1"][foreground_slice])),
                format_metric(finite_mean(metrics["iou"][foreground_slice])),
                int(metric_value(metrics["support"][foreground_slice].sum())),
            ]
        )
        writer.writerow(
            [
                "overall_accuracy",
                "",
                "",
                format_metric(metric_value(metrics["accuracy"])),
                "",
                int(metric_value(metrics["support"].sum())),
            ]
        )

    confusion_path = save_dir / "confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *class_names])
        for class_name, row in zip(class_names, confusion.tolist()):
            writer.writerow([class_name, *row])

    summary = {
        "overall_accuracy": metric_value(metrics["accuracy"]),
        "foreground_mean_f1": finite_mean(metrics["f1"][1:]),
        "foreground_mean_iou": finite_mean(metrics["iou"][1:]),
        "mean_latency_ms": mean_latency_seconds * 1000.0,
        "mean_fps": (
            1.0 / mean_latency_seconds if mean_latency_seconds > 0 else None
        ),
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def evaluate(
    model: UNet,
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
    *,
    save_dir: Path,
    size_wh: tuple[int, int],
    colormap_bgr: Sequence[Sequence[int]],
    class_names: Sequence[str],
    fourth_channel: str,
    ignore_index: int,
    strict_palette: bool,
    device: torch.device,
    amp_enabled: bool,
    warmup_steps: int,
    save_joint: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[float]]:
    num_classes = len(colormap_bgr)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    inference_times: list[float] = []
    warmed_up = False

    prediction_dir = save_dir / "masks"
    joint_dir = save_dir / "joint"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if save_joint:
        joint_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in zip(image_paths, mask_paths):
        image_bgr = read_bgr(image_path)
        mask_bgr = read_bgr(mask_path)
        mask_index = color_mask_to_index(
            mask_bgr,
            colormap_bgr,
            ignore_index=ignore_index,
            strict=strict_palette,
        )

        image_4ch = build_four_channel_image(
            image_bgr,
            fourth_channel=fourth_channel,
        )
        image_4ch, mask_index = resize_image_and_mask(
            image_4ch,
            mask_index,
            size_wh,
        )
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.transpose(image_4ch, (2, 0, 1))[None],
                dtype=np.float32,
            )
        ).to(device)

        if not warmed_up and warmup_steps > 0:
            with torch.inference_mode():
                for _ in range(warmup_steps):
                    with autocast_context(device, amp_enabled):
                        model(image_tensor)
            synchronize(device)
            warmed_up = True

        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode(), autocast_context(device, amp_enabled):
            logits, _, _ = model(image_tensor)
        synchronize(device)
        inference_times.append(time.perf_counter() - start)

        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        update_confusion_matrix(
            confusion,
            torch.from_numpy(prediction),
            torch.from_numpy(mask_index),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        name = Path(mask_path).stem
        predicted_bgr = index_to_bgr_mask(
            prediction,
            colormap_bgr,
            ignore_index=ignore_index,
        )
        write_image(prediction_dir / f"{name}.png", predicted_bgr)

        if save_joint:
            resized_image = cv2.resize(
                image_bgr,
                size_wh,
                interpolation=cv2.INTER_LINEAR,
            )
            target_bgr = index_to_bgr_mask(
                mask_index,
                colormap_bgr,
                ignore_index=ignore_index,
            )
            separator = np.full(
                (size_wh[1], 10, 3),
                255,
                dtype=np.uint8,
            )
            joint = np.concatenate(
                [
                    resized_image,
                    separator,
                    target_bgr,
                    separator,
                    predicted_bgr,
                ],
                axis=1,
            )
            write_image(joint_dir / f"{name}.png", joint)

    metrics = metrics_from_confusion(confusion)
    return confusion, metrics, inference_times


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps cannot be negative")

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    checkpoint = torch_load(checkpoint_path, device)
    model, config = build_model_from_checkpoint(checkpoint, args, device)

    dataset_path_value = args.dataset_path or config.get("dataset_path")
    if not dataset_path_value:
        raise ValueError(
            "The dataset path is not stored in the checkpoint. Pass --dataset_path."
        )
    dataset_path = Path(str(dataset_path_value)).expanduser()

    configured_size = config.get("image_size_wh", [256, 256])
    if not isinstance(configured_size, (list, tuple)) or len(configured_size) != 2:
        configured_size = [256, 256]
    image_width = args.image_width or int(configured_size[0])
    image_height = args.image_height or int(configured_size[1])
    size_wh = (image_width, image_height)

    colormap_bgr = config.get("colormap_bgr", DEFAULT_COLORMAP_BGR)
    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)
    if len(colormap_bgr) != len(class_names):
        raise ValueError("The checkpoint colormap and class names have different lengths")

    fourth_channel = (
        args.fourth_channel
        or config.get("fourth_channel")
        or "exg"
    )
    ignore_index = int(config.get("ignore_index", IGNORE_INDEX))

    test_x, test_y = load_split(dataset_path, "test")
    run_name = str(config.get("run_name", checkpoint_path.stem))
    save_dir = (
        Path(args.save_dir).expanduser()
        if args.save_dir is not None
        else Path("results") / run_name / "test"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    confusion, metrics, inference_times = evaluate(
        model,
        test_x,
        test_y,
        save_dir=save_dir,
        size_wh=size_wh,
        colormap_bgr=colormap_bgr,
        class_names=class_names,
        fourth_channel=fourth_channel,
        ignore_index=ignore_index,
        strict_palette=args.strict_palette,
        device=device,
        amp_enabled=amp_enabled,
        warmup_steps=args.warmup_steps,
        save_joint=args.save_joint,
    )

    mean_latency = float(np.mean(inference_times)) if inference_times else float("nan")
    save_metrics(
        save_dir,
        confusion,
        metrics,
        class_names,
        mean_latency,
    )

    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Evaluated images: {len(test_x)}")
    print("Class               Precision     Recall         F1        IoU")
    print("-" * 70)
    for index, class_name in enumerate(class_names):
        precision = metric_value(metrics["precision"][index])
        recall = metric_value(metrics["recall"][index])
        f1 = metric_value(metrics["f1"][index])
        iou = metric_value(metrics["iou"][index])
        print(
            f"{class_name:18s} "
            f"{format_metric(precision):>10s} "
            f"{format_metric(recall):>10s} "
            f"{format_metric(f1):>10s} "
            f"{format_metric(iou):>10s}"
        )

    print("-" * 70)
    print(
        "Foreground mean F1:",
        format_metric(finite_mean(metrics["f1"][1:])),
    )
    print(
        "Foreground mean IoU:",
        format_metric(finite_mean(metrics["iou"][1:])),
    )
    print("Overall accuracy:", format_metric(metric_value(metrics["accuracy"])))
    if np.isfinite(mean_latency) and mean_latency > 0:
        print(f"Mean latency: {mean_latency * 1000.0:.3f} ms")
        print(f"Mean FPS: {1.0 / mean_latency:.3f}")
    print(f"Results: {save_dir}")


if __name__ == "__main__":
    main()
