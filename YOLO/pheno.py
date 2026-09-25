import os
import shutil
import cv2

inputTrainImg = "/data/giorgiabartoli/phenoBench/PhenoBench/train/images"
inputTrainMasks = "/data/giorgiabartoli/phenoBench/PhenoBench/train/semantics"
inputValImg = "/data/giorgiabartoli/phenoBench/PhenoBench/val/images"
inputValMasks = "/data/giorgiabartoli/phenoBench/PhenoBench/val/semantics"

outputTrainImages = "/data/giorgiabartoli/YOLO/phenobench/images/train"
outputTrainMasks = "/data/giorgiabartoli/YOLO/phenobench/masks/train"
outputValImages = "/data/giorgiabartoli/YOLO/phenobench/images/val"
outputValMasks = "/data/giorgiabartoli/YOLO/phenobench/masks/val"

os.makedirs(outputTrainImages, exist_ok=True)
os.makedirs(outputTrainMasks, exist_ok=True)
os.makedirs(outputValImages, exist_ok=True)
os.makedirs(outputValMasks, exist_ok=True)


def copia_immagini(inputDir, outputDir):
    for nomeFile in os.listdir(inputDir):
        if nomeFile.endswith(".png"):
            shutil.copy(os.path.join(inputDir, nomeFile), os.path.join(outputDir, nomeFile))


def converti_maschere(inputDir, outputDir):
    for nomeFile in os.listdir(inputDir):
        if not nomeFile.endswith(".png"):
            continue

        path = os.path.join(inputDir, nomeFile)
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)  
        if mask is None:
            print(f"Attenzione: impossibile leggere {nomeFile}")
            continue

        mask = mask.astype("uint8")
        mask[mask == 3] = 1  # partial-crop -> mais
        mask[mask == 4] = 2  # partial-weed -> weeds

        cv2.imwrite(os.path.join(outputDir, nomeFile), mask)


print("=== copia immagini train ===")
copia_immagini(inputTrainImg, outputTrainImages)

print("=== copia immagini val ===")
copia_immagini(inputValImg, outputValImages)

print("=== conversione maschere train ===")
converti_maschere(inputTrainMasks, outputTrainMasks)

print("=== conversione maschere val ===")
converti_maschere(inputValMasks, outputValMasks)

print("Fatto.")