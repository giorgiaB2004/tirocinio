
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
from metrics import DiceLoss, DiceCELoss, FocalLoss
from model import build_unet
import matplotlib.pyplot as plt
import argparse

def load_data(dataset_path):
    train_x = sorted(glob(os.path.join(dataset_path, "train", "images", "*.png")))
    train_y = sorted(glob(os.path.join(dataset_path, "train", "masks", "*.png")))
    assert len(train_x) == len(train_y)

    valid_x = sorted(glob(os.path.join(dataset_path, "valid", "images", "*.png")))
    valid_y = sorted(glob(os.path.join(dataset_path, "valid", "masks", "*.png")))
    assert len(valid_x) == len(valid_y)

    test_x = sorted(glob(os.path.join(dataset_path, "test", "images", "*.png")))
    test_y = sorted(glob(os.path.join(dataset_path, "test", "masks", "*.png")))
    assert len(test_x) == len(test_y)

    return (train_x, train_y), (valid_x,  valid_y), (test_x, test_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, colormap, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.colormap = colormap
        self.transform = transform
        self.n_samples = len(images_path)
    
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_COLOR)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, self.size)
        mask = np.array(mask, dtype=np.uint8)

        mask_class = np.zeros(mask.shape[:2], dtype=np.uint8)
        for idx, color in enumerate(self.colormap):
            mask_class[np.all(mask == color, axis=-1)] = idx

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

    epoch_loss = epoch_loss/len(loader)
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

        epoch_loss = epoch_loss/len(loader)
        return epoch_loss
    
#aggiunto per riproducibilià
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #aggiunto per riproducibilià
    torch.use_deterministic_algorithms(True, warn_only=True) #aggiunto per riproducibilià

    parser = argparse.ArgumentParser(description="Training script per segmentazione")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()


    """ Seeding """
    seed = args.seed
    seeding(seed)


    #aggiunto per riproducibilià
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
    num_epochs = 150
    lr = 3e-4
    early_stopping_patience = 10
    checkpoint_path = f"{path}/checkpoint.pth"
    dataset_path = "/data/giorgiabartoli/unet/maisweed" #da modificare
    colormap = [ #da modificare
        [0, 0, 0],      # Background
        [0, 128, 0],    # Mais
        [0, 0, 128]     # Erbacce
    ]
    num_classes = len(colormap)

    data_str = f"Image Size: {size} - Batch Size: {batch_size} - LR: {lr} - Epochs: {num_epochs} - Num Classes: {num_classes} - "
    data_str += f"Early Stopping Patience: {early_stopping_patience} - Checkpoint Path: {checkpoint_path}"

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    # train_x, train_y = train_x[:125], train_y[:125]  # For faster training, use a subset of the training data
    data_str = f"Dataset Size: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}"
    print(data_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.2),
        A.CoarseDropout(p=0.2, max_holes=8, max_height=24, max_width=24),
    ], is_check_shapes=False) #forse da aggiungere dopo false , seed=seed

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, colormap, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, colormap, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,  #aggiunto per riproducibilià
        generator=g                  #aggiunto per riproducibilià
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,  #aggiunto per riproducibilià
        generator=g                  #aggiunto per riproducibilià
    )

    """ Model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(num_classes=num_classes)
    model = model.to(device)
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print("GPU device name:", torch.cuda.get_device_name(0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    #optimizer = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    """ Load checkpoint if exists """
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    pesi = torch.tensor([1.0, 2.21, 5.42], dtype=torch.float32).to(device)
    loss_fn = DiceCELoss(
        num_classes = num_classes,   
        weight = pesi,        
        dice_weight = 1.0,
        ce_weight = 1.0,
        ignore_index = -1
    )
    # loss_fn = nn.CrossEntropyLoss(weight=weights_tensor) 
    
    #loss_fn = FocalLoss(alpha=pesi)
    #loss_name = "focal loss"
    loss_name = "Dice + Cross Entropy Loss"
    data_str = f"Optimizer: AdamW - Loss: {loss_name}"

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
    
    plt.plot(trainLoss, label="Train Loss", color="blue", linewidth=2)
    plt.plot(valLoss, label="Validation Loss", color="red", linewidth=2)
    plt.legend(fontsize=12, loc="upper right")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig("grafico.png", dpi=300, bbox_inches="tight")
    