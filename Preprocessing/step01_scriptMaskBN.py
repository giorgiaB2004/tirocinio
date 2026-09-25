import os
import re
from collections import defaultdict
import cv2
import numpy as np
import sys


input = r"C:\Users\giorg\Desktop\Tirocinio\phenoBench\imgIniziali"
output = r"C:\Users\giorg\Desktop\Tirocinio\phenoBench\groundTruth"

os.makedirs(output, exist_ok=True)


gruppiPref = defaultdict(list)

for nomeFile in sorted(os.listdir(input)):
    nomeCompleto = os.path.join(input, nomeFile)
    nomeNoEst = os.path.splitext(nomeFile)[0]

    match = re.match(r"(task-\d+-annotation-\d+-by-\d+-tag-[a-zA-Z0-9]+)-(\d+)", nomeNoEst)

    if match:
        prefisso = match.group(1)
        gruppiPref[prefisso].append(nomeCompleto)

maschere = {}

np.set_printoptions(threshold=sys.maxsize)

for prefisso, listaFile in gruppiPref.items():

    maschera = cv2.imread(listaFile[0], 0)
    maschera = (maschera > 0).astype(np.uint8)
    
    for percorso in listaFile[1:]:
        img = cv2.imread(percorso, 0)
        maschera = ((maschera > 0) | (img > 0)).astype(np.uint8)

    maschere[prefisso] = maschera
    




gruppiTask = defaultdict(dict)

for prefisso, maschera in maschere.items():
    matchTask = re.search(r"(task-\d+)", prefisso)
    
    matchTag = re.search(r"tag-([a-zA-Z0-9]+)", prefisso)

    if matchTask and matchTag:
        idTask = matchTask.group(1)  
        nomeTag = matchTag.group(1)  
        gruppiTask[idTask][nomeTag] = maschera

for idTask, dizionarioTag in gruppiTask.items():
    dim = next(iter(dizionarioTag.values())).shape

    mask = np.zeros(dim, dtype=np.uint8)

    print(f"Task: {idTask} | Tag presenti: {list(dizionarioTag.keys())}")
    
    if "Mais" in dizionarioTag:
        mask[dizionarioTag["Mais"] == 1] = 1   # mais = 1
        
    if "Weeds" in dizionarioTag:
        mask[dizionarioTag["Weeds"] == 1] = 2  # erbacce = 2
       

    nomeFile = f"{idTask}.png"
    percorsoSalvataggio = os.path.join(output, nomeFile)
    cv2.imwrite(percorsoSalvataggio, mask)
    
    

