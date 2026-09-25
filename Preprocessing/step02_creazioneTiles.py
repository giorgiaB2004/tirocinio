import os
import re
from collections import defaultdict
import cv2
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script per il tiling di immagini e maschere")
    
    parser.add_argument("--inputDir", type=str, required=True, help="Path della cartella di input (es. groundTruth o masks)")
    parser.add_argument("--outputDir", type=str, required=True, help="Path della cartella dove salvare i tiles")
    parser.add_argument("--passo", type=int, default=256, help="Passo (stride) dello sliding window")
    parser.add_argument("--isBN", action="store_true", help="Se specificato, legge il file come maschera in scala di grigi") #da aggiungere solo per maschere in BN

    args = parser.parse_args()

    outputDir = args.outputDir
    inputDir = args.inputDir
    dim = 256
    passo = args.passo
    os.makedirs(outputDir, exist_ok=True)

    for nomeFile in sorted(os.listdir(inputDir)):
        nomeCompleto = os.path.join(inputDir, nomeFile)
        nomeNoEst = os.path.splitext(nomeFile)[0]

        if args.isBN:
            img = cv2.imread(nomeCompleto, cv2.IMREAD_UNCHANGED)  
            altezza, larghezza = img.shape 
        else:
            img = cv2.imread(nomeCompleto, cv2.IMREAD_COLOR)
            altezza, larghezza, _ = img.shape
        
        i = 0
        y = 0
        while y < altezza:
            yTile = y
            if yTile + dim > altezza:
                yTile = altezza - dim
            x = 0
            while x < larghezza:
                xTile = x
                if xTile + dim > larghezza:
                    xTile = larghezza - dim

                if args.isBN:
                    newImg = img[yTile:yTile+dim, xTile:xTile+dim]
                    newImg = newImg.astype(np.uint8)  
                else:
                    newImg = img[yTile:yTile+dim, xTile:xTile+dim]
                nome = f"{nomeNoEst}_img_{i}_passo{passo}.png"
                percorsoSalvataggio = os.path.join(outputDir, nome) 
                cv2.imwrite(percorsoSalvataggio, newImg)
                i += 1
                if xTile == larghezza - dim:
                    break
                x += passo
            if yTile == altezza - dim:
                break
            y += passo
            