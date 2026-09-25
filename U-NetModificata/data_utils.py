from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = 255
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def _index_by_stem(paths: Sequence[Path], role: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if stem in indexed:
            raise ValueError(
                f"Duplicate {role} stem '{stem}': {indexed[stem]} and {path}"
            )
        indexed[stem] = path
    return indexed


def load_split(dataset_path: str | Path, split: str) -> tuple[list[str], list[str]]:
    split_root = Path(dataset_path).expanduser() / split
    image_paths = _collect_files(split_root / "images")
    mask_paths = _collect_files(split_root / "masks")

    images_by_stem = _index_by_stem(image_paths, "image")
    masks_by_stem = _index_by_stem(mask_paths, "mask")

    missing_masks = sorted(set(images_by_stem) - set(masks_by_stem))
    missing_images = sorted(set(masks_by_stem) - set(images_by_stem))
    if missing_masks or missing_images:
        details: list[str] = []
        if missing_masks:
            details.append(f"missing masks for: {missing_masks[:10]}")
        if missing_images:
            details.append(f"missing images for: {missing_images[:10]}")
        raise ValueError(f"Invalid '{split}' split: " + "; ".join(details))

    stems = sorted(images_by_stem)
    if not stems:
        raise ValueError(f"No image-mask pairs found in: {split_root}")

    images = [str(images_by_stem[stem]) for stem in stems]
    masks = [str(masks_by_stem[stem]) for stem in stems]
    return images, masks


def load_data(
    dataset_path: str | Path,
) -> tuple[
    tuple[list[str], list[str]],
    tuple[list[str], list[str]],
    tuple[list[str], list[str]],
]:
    return (
        load_split(dataset_path, "train"),
        load_split(dataset_path, "valid"),
        load_split(dataset_path, "test"),
    )


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read: {path}")
    return image


def build_four_channel_image(
    image_bgr: np.ndarray,
    fourth_channel: str = "exg",
) -> np.ndarray:
    """Return BGR in [0, 1] plus one vegetation-related channel.

    Supported modes:
    - exg: true Excess Green computed from raw normalized BGR channels,
      scaled from [-2, 2] to [0, 1].
    - green_chromaticity: G / (R + G + B).
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 BGR image, received shape {image_bgr.shape}"
        )

    image_float = image_bgr.astype(np.float32) / 255.0
    b = image_float[..., 0]
    g = image_float[..., 1]
    r = image_float[..., 2]

    if fourth_channel == "exg":
        exg = 2.0 * g - r - b
        extra = np.clip((exg + 2.0) / 4.0, 0.0, 1.0)
    elif fourth_channel == "green_chromaticity":
        denominator = np.maximum(r + g + b, 1e-8)
        extra = g / denominator
    else:
        raise ValueError(
            "fourth_channel must be 'exg' or 'green_chromaticity', "
            f"received: {fourth_channel!r}"
        )

    return np.ascontiguousarray(
        np.concatenate([image_float, extra[..., None]], axis=-1),
        dtype=np.float32,
    )


def _validate_colormap(colormap_bgr: Sequence[Sequence[int]]) -> np.ndarray:
    palette = np.asarray(colormap_bgr, dtype=np.int64)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("colormap_bgr must have shape [num_classes, 3]")
    if np.any(palette < 0) or np.any(palette > 255):
        raise ValueError("Every colormap value must be in [0, 255]")
    if len({tuple(color) for color in palette.tolist()}) != len(palette):
        raise ValueError("colormap_bgr contains duplicate colors")
    return palette.astype(np.uint8)


def color_mask_to_index(
    mask_bgr: np.ndarray,
    colormap_bgr: Sequence[Sequence[int]],
    *,
    ignore_index: int = IGNORE_INDEX,
    strict: bool = True,
) -> np.ndarray:
    if mask_bgr.ndim != 3 or mask_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 BGR mask, received shape {mask_bgr.shape}"
        )

    palette = _validate_colormap(colormap_bgr)
    if 0 <= ignore_index < len(palette):
        raise ValueError("ignore_index must not overlap a valid class index")

    indexed = np.full(mask_bgr.shape[:2], ignore_index, dtype=np.uint8)
    for class_id, color in enumerate(palette):
        indexed[np.all(mask_bgr == color, axis=-1)] = class_id

    unknown = indexed == ignore_index
    if strict and np.any(unknown):
        unknown_colors, counts = np.unique(
            mask_bgr[unknown].reshape(-1, 3),
            axis=0,
            return_counts=True,
        )
        examples = [
            {"bgr": color.tolist(), "pixels": int(count)}
            for color, count in zip(unknown_colors[:10], counts[:10])
        ]
        raise ValueError(
            f"Mask contains {int(unknown.sum())} pixels outside the BGR palette. "
            f"Examples: {examples}"
        )

    return indexed


def index_to_bgr_mask(
    mask: np.ndarray,
    colormap_bgr: Sequence[Sequence[int]],
    *,
    ignore_index: int = IGNORE_INDEX,
    ignore_color_bgr: Sequence[int] = (255, 0, 255),
) -> np.ndarray:
    palette = _validate_colormap(colormap_bgr)
    if mask.ndim != 2:
        raise ValueError(f"Expected an HxW indexed mask, received shape {mask.shape}")

    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        output[mask == class_id] = color
    output[mask == ignore_index] = np.asarray(ignore_color_bgr, dtype=np.uint8)
    return output


def resize_image_and_mask(
    image_4ch: np.ndarray,
    mask_index: np.ndarray,
    size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image_resized = cv2.resize(
        image_4ch,
        size_wh,
        interpolation=cv2.INTER_LINEAR,
    )
    mask_resized = cv2.resize(
        mask_index,
        size_wh,
        interpolation=cv2.INTER_NEAREST,
    )
    return (
        np.ascontiguousarray(image_resized, dtype=np.float32),
        np.ascontiguousarray(mask_resized, dtype=np.uint8),
    )


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        size_wh: tuple[int, int],
        colormap_bgr: Sequence[Sequence[int]],
        *,
        fourth_channel: str = "exg",
        color_transform: Any | None = None,
        spatial_transform: Any | None = None,
        strict_palette: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Received {len(image_paths)} images and {len(mask_paths)} masks"
            )
        if not image_paths:
            raise ValueError("The dataset is empty")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.size_wh = size_wh
        self.colormap_bgr = [list(color) for color in colormap_bgr]
        self.fourth_channel = fourth_channel
        self.color_transform = color_transform
        self.spatial_transform = spatial_transform
        self.strict_palette = strict_palette
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_bgr = read_bgr(self.image_paths[index])
        mask_bgr = read_bgr(self.mask_paths[index])

        mask_index = color_mask_to_index(
            mask_bgr,
            self.colormap_bgr,
            ignore_index=self.ignore_index,
            strict=self.strict_palette,
        )

        if self.color_transform is not None:
            image_bgr = self.color_transform(image=image_bgr)["image"]

        image_4ch = build_four_channel_image(
            image_bgr,
            fourth_channel=self.fourth_channel,
        )

        if self.spatial_transform is not None:
            augmented = self.spatial_transform(image=image_4ch, mask=mask_index)
            image_4ch = augmented["image"]
            mask_index = augmented["mask"]

        image_4ch, mask_index = resize_image_and_mask(
            image_4ch,
            mask_index,
            self.size_wh,
        )

        image_chw = np.ascontiguousarray(
            np.transpose(image_4ch, (2, 0, 1)),
            dtype=np.float32,
        )
        mask_long = np.ascontiguousarray(mask_index, dtype=np.int64)

        return torch.from_numpy(image_chw), torch.from_numpy(mask_long)


def update_confusion_matrix(
    confusion: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    prediction = prediction.reshape(-1).to(torch.long)
    target = target.reshape(-1).to(torch.long)

    invalid_target = (target != ignore_index) & (
        (target < 0) | (target >= num_classes)
    )
    if torch.any(invalid_target):
        invalid_values = torch.unique(target[invalid_target]).detach().cpu().tolist()
        raise ValueError(f"Invalid target class values: {invalid_values}")

    valid = target != ignore_index
    target = target[valid]
    prediction = prediction[valid]

    if target.numel() == 0:
        return confusion
    if torch.any((prediction < 0) | (prediction >= num_classes)):
        invalid_values = torch.unique(
            prediction[(prediction < 0) | (prediction >= num_classes)]
        ).detach().cpu().tolist()
        raise ValueError(f"Invalid predicted class values: {invalid_values}")

    indices = target * num_classes + prediction
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).to(confusion.device)
    return confusion


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    nan = torch.full_like(numerator, torch.nan, dtype=torch.float64)
    return torch.where(
        denominator > 0,
        numerator.to(torch.float64) / denominator.to(torch.float64),
        nan,
    )


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
    matrix = confusion.to(torch.float64)
    true_positive = torch.diag(matrix)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    false_positive = predicted - true_positive
    false_negative = support - true_positive

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    iou = _safe_divide(
        true_positive,
        true_positive + false_positive + false_negative,
    )
    dice = _safe_divide(
        2.0 * true_positive,
        2.0 * true_positive + false_positive + false_negative,
    )
    accuracy = _safe_divide(true_positive.sum()[None], matrix.sum()[None])[0]

    return {
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "iou": iou,
        "support": support,
        "accuracy": accuracy,
    }
