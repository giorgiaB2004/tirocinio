import os
import re
import shutil

"""
    dataset_path/
      ├── train/
      │     ├── images/
      │     └── masks/
      ├── valid/
      │     ├── images/
      │     └── masks/
      └── test/
            ├── images/
            └── masks/
    """

inputMasks = r"/data/giorgiabartoli/modificheUNET/groundTruthTiles64"
inputImg = r"/data/giorgiabartoli/modificheUNET/rawTiles64"

outputTrainImages = r"/data/giorgiabartoli/modificheUNET/maisweed64/train/images"
outputTrainMasks = r"/data/giorgiabartoli/modificheUNET/maisweed64/train/masks"

outputValidImages = r"/data/giorgiabartoli/modificheUNET/maisweed64/valid/images"
outputValidMasks = r"/data/giorgiabartoli/modificheUNET/maisweed64/valid/masks"

outputTestImages = r"/data/giorgiabartoli/modificheUNET/maisweed64/test/images"
outputTestMasks = r"/data/giorgiabartoli/modificheUNET/maisweed64/test/masks"

os.makedirs(outputTrainImages, exist_ok=True)
os.makedirs(outputTrainMasks, exist_ok=True)

os.makedirs(outputValidImages, exist_ok=True)
os.makedirs(outputValidMasks, exist_ok=True)

os.makedirs(outputTestImages, exist_ok=True)
os.makedirs(outputTestMasks, exist_ok=True)


def get_destinazione(nomeFile, isMask=False):

    match = re.search(r"task-(\d+)", nomeFile)
    if not match:
        return None 
        
    task = int(match.group(1))
    
    # Definizione dei range (estremi inclusi)
    if (4 <= task <= 25) or (40 <= task <= 68):
        return outputTrainMasks if isMask else outputTrainImages
    elif 69 <= task <= 78:
        return outputValidMasks if isMask else outputValidImages
    elif 79 <= task <= 88:
        return outputTestMasks if isMask else outputTestImages
    
    return None 


for nomeFile in os.listdir(inputImg):
    if nomeFile.endswith(".png"):
        dest = get_destinazione(nomeFile, isMask=False)
        if dest:
            shutil.copy(os.path.join(inputImg, nomeFile), os.path.join(dest, nomeFile))

for nomeFile in os.listdir(inputMasks):
    if nomeFile.endswith(".png"):
        dest = get_destinazione(nomeFile, isMask=True)
        if dest:
            shutil.copy(os.path.join(inputMasks, nomeFile), os.path.join(dest, nomeFile))