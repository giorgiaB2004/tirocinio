import os
import random
import time
import datetime
import numpy as np
import pandas as pd
import albumentations as A
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import seeding, create_dir, shuffling, epoch_time, EarlyStopping, save_checkpoint, load_checkpoint
from metrics import DiceLoss, DiceCELoss
from model import build_unet
import matplotlib.pyplot as plt
import argparse

def load_data(dataset_path):
    train_x = sorted(glob(os.path.join(dataset_path, "train", "images", "*.png")))
    train_y = sorted(glob(os.path.join(dataset_path, "train", "masks", "*.png")))
    assert len(train_x) == len(train_y), f"Errore: Trovate {len(train_x)} immagini ma {len(train_y)} maschere in train."

    valid_x = sorted(glob(os.path.join(dataset_path, "valid", "images", "*.png")))
    valid_y = sorted(glob(os.path.join(dataset_path, "valid", "masks", "*.png")))
    assert len(valid_x) == len(valid_y), f"Errore: Trovate {len(valid_x)} immagini ma {len(valid_y)} maschere in valid."

    test_x = sorted(glob(os.path.join(dataset_path, "test", "images", "*.png")))
    test_y = sorted(glob(os.path.join(dataset_path, "test", "masks", "*.png")))
    
    if len(test_y) == 0:
        print("[INFO] Nessuna maschera di riferimento trovata per il test. Verrà eseguita solo l'inferenza.")
        test_y = None
    else:
        assert len(test_x) == len(test_y), f"Errore: Trovate {len(test_x)} immagini ma {len(test_y)} maschere in test."

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask[mask == 3] = 1  
        mask[mask == 4] = 2  
        
        image = cv2.resize(image, self.size)
        
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        
        mask_class = np.array(mask, dtype=np.uint8)

        return image, mask_class

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        with torch.cuda.amp.autocast():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
        return epoch_loss
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    parser = argparse.ArgumentParser(description="Training script per segmentazione")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()

    """ Seeding """
    seed = args.seed
    seeding(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    """ Directories """
    path = "files"
    create_dir(path)

    """ Hyperparameters """
    image_w = 256
    image_h = 256
    size = (image_w, image_h)
    batch_size = 16
    start_epoch = 0
    num_epochs = 500
    lr = 3e-4 
    early_stopping_patience = 50
    checkpoint_path = f"{path}/checkpoint.pth"
    dataset_path = "/data/giorgiabartoli/phenoBenchUnet/dataset" #da modificare
    
    num_classes = 3

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    print(f"Dataset Size: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.2),
        A.CoarseDropout(p=0.2, max_holes=8, max_height=24, max_width=24),
    ], is_check_shapes=False)

    """ Dataset and loader """
    
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    """ Model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(num_classes=num_classes)
    model = model.to(device)
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print("GPU device name:", torch.cuda.get_device_name(0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    """ Load checkpoint if exists """
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

   
    pesi = torch.tensor([0.3, 4.0, 18.0], dtype=torch.float32).to(device)
    loss_fn = DiceCELoss(
        num_classes = num_classes,   
        weight = pesi,        
        dice_weight = 1.0,
        ce_weight = 1.0,
        ignore_index = -1
    )
    
    trainLoss = []
    valLoss = []

    """ Training the model """
    for epoch in range(start_epoch+1, num_epochs, 1):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        trainLoss.append(train_loss)
        valLoss.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"[{epoch:02}/{num_epochs:02}] | Epoch Time: {epoch_mins}m {epoch_secs}s - Train Loss: {train_loss:.4f} - Val. Loss: {valid_loss:.4f}"
        print(data_str)

        scheduler.step(valid_loss)
        early_stopping(valid_loss, model, optimizer, epoch, checkpoint_path)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training will stop.")
            break
    
    plt.figure()
    plt.plot(trainLoss, label="Train Loss", color="blue", linewidth=2)
    plt.plot(valLoss, label="Validation Loss", color="red", linewidth=2)
    plt.legend(fontsize=12, loc="upper right")
    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig("grafico.png", dpi=300, bbox_inches="tight")