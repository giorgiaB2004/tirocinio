from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore[assignment]

from data_utils import (
    IGNORE_INDEX,
    SegmentationDataset,
    load_data,
    metrics_from_confusion,
    update_confusion_matrix,
)
from metrics import CombinedSegmentationLoss, DiceCELoss, SIGReg
from model import UNet, build_unet


COLORMAP_BGR = [
    [0, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
]
CLASS_NAMES = ["background", "mais", "weeds"]
DEFAULT_CLASS_WEIGHTS = [1.0, 2.21, 5.42]


class SegmentationOnlyLoss(nn.Module):
    def __init__(self, segmentation_loss: nn.Module) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss

    def forward(
        self,
        logits: torch.Tensor,
        bottleneck: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        del bottleneck
        return self.segmentation_loss(logits, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a four-channel U-Net for semantic segmentation"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/giorgiabartoli/modificheUNET/maisweed",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="files")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=0.0)

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
        metavar=("RATE1", "RATE2", "RATE3"),
    )
    parser.add_argument(
        "--aspp_separable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min_attention", type=float, default=0.25)
    parser.add_argument("--attention_aux_weight", type=float, default=0.1)
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=3,
        default=DEFAULT_CLASS_WEIGHTS,
        metavar=("BACKGROUND", "MAIS", "WEEDS"),
    )

    parser.add_argument(
        "--fourth_channel",
        type=str,
        default="exg",
        choices=["exg", "green_chromaticity"],
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict_palette",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--sigreg_knots", type=int, default=17)
    parser.add_argument("--sigreg_slices", type=int, default=256)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = [
        "image_width",
        "image_height",
        "batch_size",
        "epochs",
        "base_channels",
        "aspp_branch_channels",
        "patience",
        "scheduler_patience",
    ]
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay cannot be negative")
    if args.lambda_reg < 0:
        raise ValueError("--lambda_reg cannot be negative")
    if args.attention_aux_weight < 0:
        raise ValueError("--attention_aux_weight cannot be negative")
    if args.grad_clip < 0:
        raise ValueError("--grad_clip cannot be negative")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms() -> tuple[Any, Any]:
    if A is None:
        raise ImportError(
            "Albumentations is required for augmentation. Install it or use "
            "--no-augment."
        )

    color_transform = A.Compose([A.RandomBrightnessContrast(p=0.2)])

    shift_kwargs: dict[str, Any] = {
        "shift_limit": 0.05,
        "scale_limit": 0.10,
        "rotate_limit": 10,
        "interpolation": cv2.INTER_LINEAR,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 0.5,
    }
    parameters = inspect.signature(A.ShiftScaleRotate).parameters
    if "mask_interpolation" in parameters:
        shift_kwargs["mask_interpolation"] = cv2.INTER_NEAREST
    if "fill" in parameters:
        shift_kwargs["fill"] = 0
    elif "value" in parameters:
        shift_kwargs["value"] = 0
    if "fill_mask" in parameters:
        shift_kwargs["fill_mask"] = IGNORE_INDEX
    elif "mask_value" in parameters:
        shift_kwargs["mask_value"] = IGNORE_INDEX

    spatial_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(**shift_kwargs),
        ],
        is_check_shapes=False,
    )
    return color_transform, spatial_transform


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


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


def attention_auxiliary_loss(
    attention_logits: Sequence[torch.Tensor],
    target: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for logits in attention_logits:
        if target.shape[-2:] == logits.shape[-2:]:
            resized_target = target
        else:
            resized_target = F.interpolate(
                target.unsqueeze(1).to(torch.float32),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).to(torch.long)

        if not torch.any(resized_target != ignore_index):
            continue
        losses.append(
            F.cross_entropy(
                logits,
                resized_target,
                weight=class_weights,
                ignore_index=ignore_index,
            )
        )

    if not losses:
        return target.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def forward_and_loss(
    model: UNet,
    inputs: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module,
    class_weights: torch.Tensor,
    attention_aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, bottleneck, auxiliary_logits = model(inputs)
    primary_loss = loss_fn(logits, bottleneck, target)
    auxiliary_loss = attention_auxiliary_loss(
        auxiliary_logits,
        target,
        class_weights,
        ignore_index=IGNORE_INDEX,
    )
    total_loss = primary_loss + attention_aux_weight * auxiliary_loss
    return logits, total_loss, primary_loss, auxiliary_loss


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    class_weights: torch.Tensor,
    device: torch.device,
    scaler: Any,
    *,
    amp_enabled: bool,
    attention_aux_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "primary_loss": 0.0, "attention_loss": 0.0}
    samples = 0

    for inputs, target in loader:
        inputs = inputs.to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        target = target.to(
            device,
            dtype=torch.long,
            non_blocking=device.type == "cuda",
        )
        batch_size = inputs.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            _, total_loss, primary_loss, auxiliary_loss = forward_and_loss(
                model,
                inputs,
                target,
                loss_fn,
                class_weights,
                attention_aux_weight,
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite training loss: {total_loss.item()}")

        scaler.scale(total_loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(total_loss.detach()) * batch_size
        totals["primary_loss"] += float(primary_loss.detach()) * batch_size
        totals["attention_loss"] += float(auxiliary_loss.detach()) * batch_size
        samples += batch_size

    if samples == 0:
        raise ValueError("Training loader is empty")
    return {name: value / samples for name, value in totals.items()}


@torch.inference_mode()
def validate_one_epoch(
    model: UNet,
    loader: DataLoader,
    loss_fn: nn.Module,
    class_weights: torch.Tensor,
    device: torch.device,
    *,
    amp_enabled: bool,
    attention_aux_weight: float,
    num_classes: int,
) -> tuple[dict[str, float], torch.Tensor, dict[str, torch.Tensor]]:
    model.eval()
    totals = {"loss": 0.0, "primary_loss": 0.0, "attention_loss": 0.0}
    samples = 0
    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )

    for inputs, target in loader:
        inputs = inputs.to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        target = target.to(
            device,
            dtype=torch.long,
            non_blocking=device.type == "cuda",
        )
        batch_size = inputs.shape[0]

        with autocast_context(device, amp_enabled):
            logits, total_loss, primary_loss, auxiliary_loss = forward_and_loss(
                model,
                inputs,
                target,
                loss_fn,
                class_weights,
                attention_aux_weight,
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite validation loss: {total_loss.item()}"
            )

        totals["loss"] += float(total_loss.detach()) * batch_size
        totals["primary_loss"] += float(primary_loss.detach()) * batch_size
        totals["attention_loss"] += float(auxiliary_loss.detach()) * batch_size
        samples += batch_size

        prediction = logits.argmax(dim=1)
        update_confusion_matrix(
            confusion,
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
        )

    if samples == 0:
        raise ValueError("Validation loader is empty")

    averaged = {name: value / samples for name, value in totals.items()}
    confusion_cpu = confusion.cpu()
    return averaged, confusion_cpu, metrics_from_confusion(confusion_cpu)


def nanmean_or_nan(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean())


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_checkpoint(
    destination: Path,
    *,
    epoch: int,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Any,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "config": config,
        },
        destination,
    )


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")
    return checkpoint


def resume_training(
    checkpoint_path: Path,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Any,
    device: torch.device,
) -> tuple[int, float, int, list[dict[str, float]]]:
    checkpoint = torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    completed_epochs = int(checkpoint.get("epoch", 0))
    best_loss = float(checkpoint.get("best_validation_loss", float("inf")))
    stale_epochs = int(checkpoint.get("epochs_without_improvement", 0))
    history = list(checkpoint.get("history", []))
    return completed_epochs, best_loss, stale_epochs, history


def save_history_csv(history: list[dict[str, float]], destination: Path) -> None:
    if not history:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_plots(
    history: list[dict[str, float]],
    result_dir: Path,
    class_names: Sequence[str],
) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    plt.plot(epochs, [row["valid_loss"] for row in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "loss_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for class_name in class_names:
        key = f"iou_{class_name}"
        plt.plot(epochs, [row[key] for row in history], label=f"IoU {class_name}")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.ylim(0.0, 1.05)
    plt.title("Validation IoU per class")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "iou_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_loss(
    num_classes: int,
    class_weights: torch.Tensor,
    device: torch.device,
    *,
    lambda_reg: float,
    sigreg_knots: int,
    sigreg_slices: int,
) -> nn.Module:
    segmentation_loss = DiceCELoss(
        num_classes=num_classes,
        weight=class_weights,
        dice_weight=1.0,
        ce_weight=1.0,
        ignore_index=IGNORE_INDEX,
    )

    if lambda_reg == 0:
        return SegmentationOnlyLoss(segmentation_loss).to(device)

    regularization_loss = SIGReg(
        knots=sigreg_knots,
        num_slices=sigreg_slices,
    )
    return CombinedSegmentationLoss(
        seg_loss_fn=segmentation_loss,
        reg_loss_fn=regularization_loss,
        lambda_reg=lambda_reg,
    ).to(device)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed, args.deterministic)

    num_classes = len(COLORMAP_BGR)
    if len(CLASS_NAMES) != num_classes:
        raise ValueError("CLASS_NAMES and COLORMAP_BGR have different lengths")
    if len(args.class_weights) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class weights, received {len(args.class_weights)}"
        )

    size_wh = (args.image_width, args.image_height)
    dataset_path = Path(args.dataset_path).expanduser()
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(
        dataset_path
    )
    print(
        f"Dataset size - train: {len(train_x)}, valid: {len(valid_x)}, "
        f"test: {len(test_x)}"
    )

    if args.augment:
        color_transform, spatial_transform = build_transforms()
    else:
        color_transform, spatial_transform = None, None

    train_dataset = SegmentationDataset(
        train_x,
        train_y,
        size_wh,
        COLORMAP_BGR,
        fourth_channel=args.fourth_channel,
        color_transform=color_transform,
        spatial_transform=spatial_transform,
        strict_palette=args.strict_palette,
        ignore_index=IGNORE_INDEX,
    )
    valid_dataset = SegmentationDataset(
        valid_x,
        valid_y,
        size_wh,
        COLORMAP_BGR,
        fourth_channel=args.fourth_channel,
        strict_palette=args.strict_palette,
        ignore_index=IGNORE_INDEX,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    pin_memory = device.type == "cuda"

    train_generator = torch.Generator().manual_seed(args.seed)
    valid_generator = torch.Generator().manual_seed(args.seed + 1)
    common_loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        drop_last=False,
        **common_loader_args,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=valid_generator,
        drop_last=False,
        **common_loader_args,
    )

    model = build_unet(
        num_classes=num_classes,
        in_channels=4,
        attention_type=args.attention_type,
        class_weights=args.class_weights,
        dropout_rate=args.dropout_rate,
        use_aspp=args.use_aspp,
        base_channels=args.base_channels,
        aspp_branch_channels=args.aspp_branch_channels,
        aspp_rates=args.aspp_rates,
        aspp_separable=args.aspp_separable,
        min_attention=args.min_attention,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
    )
    scaler = create_grad_scaler(amp_enabled)
    class_weights = torch.tensor(
        args.class_weights,
        dtype=torch.float32,
        device=device,
    )
    loss_fn = build_loss(
        num_classes,
        class_weights,
        device,
        lambda_reg=args.lambda_reg,
        sigreg_knots=args.sigreg_knots,
        sigreg_slices=args.sigreg_slices,
    )

    aspp_label = "aspp" if args.use_aspp else "noaspp"
    generated_run_name = (
        f"{args.attention_type}_{aspp_label}_drop{args.dropout_rate:g}_"
        f"{args.fourth_channel}_seed{args.seed}"
    )
    run_name = args.run_name or generated_run_name
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    result_dir = Path(args.results_dir).expanduser() / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / f"checkpoint_{run_name}.pth"
    last_checkpoint_path = checkpoint_dir / f"checkpoint_{run_name}_last.pth"

    config: dict[str, Any] = {
        "run_name": run_name,
        "dataset_path": str(dataset_path.resolve()),
        "image_size_wh": list(size_wh),
        "colormap_bgr": COLORMAP_BGR,
        "class_names": CLASS_NAMES,
        "class_weights": [float(value) for value in args.class_weights],
        "fourth_channel": args.fourth_channel,
        "ignore_index": IGNORE_INDEX,
        "model": model.config,
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "attention_aux_weight": args.attention_aux_weight,
            "lambda_reg": args.lambda_reg,
            "seed": args.seed,
        },
    }
    with (result_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    start_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        (
            start_epoch,
            best_validation_loss,
            epochs_without_improvement,
            history,
        ) = resume_training(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        print(f"Resumed from epoch {start_epoch}: {resume_path}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; AMP: {amp_enabled}; parameters: {parameter_count:,}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        start_time = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            class_weights,
            device,
            scaler,
            amp_enabled=amp_enabled,
            attention_aux_weight=args.attention_aux_weight,
            grad_clip=args.grad_clip,
        )
        valid_metrics, confusion, segmentation_metrics = validate_one_epoch(
            model,
            valid_loader,
            loss_fn,
            class_weights,
            device,
            amp_enabled=amp_enabled,
            attention_aux_weight=args.attention_aux_weight,
            num_classes=num_classes,
        )

        scheduler.step(valid_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        foreground_miou = nanmean_or_nan(segmentation_metrics["iou"][1:])

        history_row: dict[str, float] = {
            "epoch": float(epoch_number),
            "train_loss": train_metrics["loss"],
            "train_primary_loss": train_metrics["primary_loss"],
            "train_attention_loss": train_metrics["attention_loss"],
            "valid_loss": valid_metrics["loss"],
            "valid_primary_loss": valid_metrics["primary_loss"],
            "valid_attention_loss": valid_metrics["attention_loss"],
            "foreground_miou": foreground_miou,
            "learning_rate": current_lr,
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            history_row[f"iou_{class_name}"] = float(
                segmentation_metrics["iou"][class_index]
            )
        history.append(history_row)

        improved = valid_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = valid_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint_path,
            epoch=epoch_number,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            config=config,
        )
        if improved:
            save_checkpoint(
                best_checkpoint_path,
                epoch=epoch_number,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                config=config,
            )

        save_history_csv(history, result_dir / "history.csv")
        elapsed = time.perf_counter() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"[{epoch_number:03d}/{args.epochs:03d}] "
            f"{minutes}m {seconds:02d}s | "
            f"train {train_metrics['loss']:.4f} | "
            f"valid {valid_metrics['loss']:.4f} | "
            f"foreground mIoU {foreground_miou:.4f} | "
            f"lr {current_lr:.3e}"
        )
        print("Validation confusion matrix:\n", confusion.numpy())

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs "
                "without validation-loss improvement."
            )
            break

    save_history_csv(history, result_dir / "history.csv")
    save_plots(history, result_dir, CLASS_NAMES)
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
