import argparse
import gc
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from model import build_unet

parser = argparse.ArgumentParser(description="Generazione LayerCAM per U-Net")

parser.add_argument(
    "--test_dir", 
    type=str, 
    required=True,
    help="Percorso alla cartella delle immagini di test"
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    required=True,
    help="Percorso alla cartella di output per salvare le heatmap"
)
parser.add_argument(
    "--checkpoint_path", 
    type=str, 
    required=True,
    help="Percorso al file .pth del checkpoint"
)
parser.add_argument(
    "--target_layer", 
    type=str, 
    default="d4.conv",
    help="Target layer del modello per LayerCAM (es. d4.conv)"
)

args = parser.parse_args()

TEST_DIR = args.test_dir
OUTPUT_DIR = args.output_dir
CHECKPOINT_PATH = args.checkpoint_path
TARGET_LAYER = args.target_layer

DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
print(f"Esecuzione su dispositivo: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_extra_channel(rgb_tensor, channel_type="exg"):
    R = rgb_tensor[:, 0:1, :, :]
    G = rgb_tensor[:, 1:2, :, :]
    B = rgb_tensor[:, 2:3, :, :]
    
    if channel_type == "exg":
        extra = 2.0 * G - R - B
    elif channel_type in ["green_chromaticity", "gcc"]:
        eps = 1e-7
        extra = G / (R + G + B + eps)
    else:
        raise ValueError(f"Tipo di canale non supportato: {channel_type}")
        
    return extra

ckpt_str = CHECKPOINT_PATH.lower()
if "green_chromaticity" in ckpt_str or "gc" in ckpt_str:
    EXTRA_CHANNEL_TYPE = "green_chromaticity"
else:
    EXTRA_CHANNEL_TYPE = "exg"


checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model_config = checkpoint["config"]["model"]
class_names = checkpoint["config"].get("class_names", ["background", "mais", "weeds"])


print("\nConfigurazione del modello:")
print(model_config)

model = build_unet(**model_config).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
cam_extractor = LayerCAM(model, target_layer=TARGET_LAYER)

CLASS_TARGETS = {1: class_names[1], 2: class_names[2]}
targets_list = list(CLASS_TARGETS.items())

transform = transforms.ToTensor()

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
image_paths = [p for p in Path(TEST_DIR).glob("*") if p.suffix.lower() in image_extensions]


for idx, img_path in enumerate(image_paths):
    raw_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    if model_config.get("in_channels") == 4 and input_tensor.shape[1] == 3:
        fourth_channel = compute_extra_channel(input_tensor, channel_type=EXTRA_CHANNEL_TYPE)
        input_tensor = torch.cat([input_tensor, fourth_channel], dim=1)

    input_tensor.requires_grad_(True)
    logits, _, _ = model(input_tensor)
    
    class_scores = logits.sum(dim=(-2, -1))

    for i, (class_idx, class_label) in enumerate(targets_list):
        is_last = (i == len(targets_list) - 1)
        
        activation_maps = cam_extractor(
            class_idx=class_idx, 
            scores=class_scores, 
            retain_graph=not is_last
        )
        
        if activation_maps and len(activation_maps) > 0:
            act_map = activation_maps[0].cpu().squeeze()
            
            act_map_min, act_map_max = act_map.min(), act_map.max()
            if act_map_max > act_map_min:
                act_map = (act_map - act_map_min) / (act_map_max - act_map_min)
            else:
                act_map = torch.zeros_like(act_map)
            
            heatmap_pil = transforms.ToPILImage(mode='F')(act_map)
            result_img = overlay_mask(raw_img, heatmap_pil, alpha=0.5)
            
            save_name = f"{img_path.stem}_{class_label}_layercam.png"
            result_img.save(os.path.join(OUTPUT_DIR, save_name))

    del logits, class_scores, input_tensor, raw_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

cam_extractor.remove_hooks()

print(f"\nElaborazione completata! Heatmap salvate in: {OUTPUT_DIR}")