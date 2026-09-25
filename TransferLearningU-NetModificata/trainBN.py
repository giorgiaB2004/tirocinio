from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from train import SegmentationOnlyLoss

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import (
    IGNORE_INDEX,
    SegmentationDataset,
    load_split,
    metrics_from_confusion,
    update_confusion_matrix,
    SafePhenoDataset,
)
from metrics import CombinedSegmentationLoss, DiceCELoss, SIGReg
from model import build_unet
from train import (
    autocast_context,
    create_grad_scaler,
    forward_and_loss,
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    save_history_csv,
    save_plots,
    seed_everything,
)

# Definizione dei colori standard per PhenoBench (solitamente coerenti con mais/weed)
# Modifica questa costante nel tuo script di addestramento
COLORMAP_BGR = [
    [0, 0, 0],  # Cerca il pixel [0,0,0] e assegna la classe 0 (Sfondo)
    [1, 1, 1],  # Cerca il pixel [1,1,1] e assegna la classe 1 (Mais)
    [2, 2, 2]   # Cerca il pixel [2,2,2] e assegna la classe 2 (Erbacce)
]
CLASS_NAMES = ["background", "crop", "weed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train U-Net on PhenoBench")
    parser.add_argument("--dataset_path", type=str, required=True, help="Percorso del dataset PhenoBench")
    parser.add_argument("--checkpoint_dir", type=str, default="pretrained_files")
    parser.add_argument("--results_dir", type=str, default="pretrained_results")
    parser.add_argument("--run_name", type=str, default="phenobench_pretrain")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--attention_type", type=str, default="none", choices=["none", "class_aware", "dual", "both"])
    parser.add_argument("--use_aspp", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--base_channels", type=int, default=64)
    
    parser.add_argument("--fourth_channel", type=str, default="exg", choices=["exg", "green_chromaticity"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, args.deterministic)

    num_classes = len(COLORMAP_BGR)
    size_wh = (args.image_width, args.image_height)
    
    # Carica PhenoBench usando i percorsi strutturati (es. train/ e val/)
    # Nota: Assicurati che load_data o una funzione analoga punti correttamente a train e val
    dataset_path = Path(args.dataset_path).expanduser()
    train_x, train_y = load_split(dataset_path, "train")
    valid_x, valid_y = load_split(dataset_path, "valid")
    
    print(f"PhenoBench - Train: {len(train_x)}, Validation: {len(valid_x)}")

    train_dataset = SafePhenoDataset(
        train_x, train_y, size_wh, COLORMAP_BGR,
        fourth_channel=args.fourth_channel,
        strict_palette=False,  # PhenoBench potrebbe avere sfumature o formati diversi
        ignore_index=IGNORE_INDEX
    )
    valid_dataset = SafePhenoDataset(
        valid_x, valid_y, size_wh, COLORMAP_BGR,
        fourth_channel=args.fourth_channel,
        strict_palette=False,
        ignore_index=IGNORE_INDEX
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Costruisci l'architettura
    model = build_unet(
        num_classes=num_classes,
        in_channels=4,
        attention_type=args.attention_type,
        dropout_rate=args.dropout_rate,
        use_aspp=args.use_aspp,
        base_channels=args.base_channels,
    ).to(device)

    # Ottimizzatore standard per pre-training (addestriamo TUTTI i pesi fin dall'inizio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    scaler = create_grad_scaler(amp_enabled)

    # Loss standard per segmentazione
    class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)  # Bilanciato inizialmente
    base_seg_loss = DiceCELoss(num_classes=num_classes, weight=class_weights, ignore_index=IGNORE_INDEX).to(device)
    loss_fn = SegmentationOnlyLoss(base_seg_loss)
    #loss_fn = DiceCELoss(num_classes=num_classes, weight=class_weights, ignore_index=IGNORE_INDEX).to(device)

    # Configurazione di salvataggio
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    result_dir = Path(args.results_dir).expanduser() / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / f"pretrained_{args.run_name}.pth"

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print(f"Inizio pre-addestramento su PhenoBench ({args.epochs} epoche)...")
    for epoch_index in range(args.epochs):
        epoch_number = epoch_index + 1
        start_time = time.perf_counter()

        # Addestramento ed Eval standard
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, class_weights, device, scaler,
            amp_enabled=amp_enabled, attention_aux_weight=0.0, grad_clip=0.0
        )
        valid_metrics, _, segmentation_metrics = validate_one_epoch(
            model, valid_loader, loss_fn, class_weights, device,
            amp_enabled=amp_enabled, attention_aux_weight=0.0, num_classes=num_classes
        )

        scheduler.step(valid_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        foreground_miou = float(segmentation_metrics["iou"][1:].nanmean())

        history_row = {
            "epoch": float(epoch_number),
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "foreground_miou": foreground_miou,
            "learning_rate": current_lr,
        }
        history.append(history_row)

        improved = valid_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = valid_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path, epoch=epoch_number, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, best_validation_loss=best_validation_loss,
                epochs_without_improvement=0, history=history, config={}
            )
        else:
            epochs_without_improvement += 1

        elapsed = time.perf_counter() - start_time
        print(f"Epoch {epoch_number:03d} | Train: {train_metrics['loss']:.4f} | Val: {valid_metrics['loss']:.4f} | mIoU: {foreground_miou:.4f} | Tempo: {elapsed:.1f}s")

        if epochs_without_improvement >= args.patience:
            print("Early stopping nel pre-addestramento.")
            break

    save_history_csv(history, result_dir / "pretrain_history.csv")
    print(f"Pre-addestramento completato! Modello migliore salvato in: {best_checkpoint_path}")


if __name__ == "__main__":
    main()