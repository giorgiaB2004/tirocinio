import os
import re
from collections import defaultdict
import cv2
import numpy as np


input = r"/data/giorgiabartoli/modificheUNET/imgIniziali"
output = r"/data/giorgiabartoli/modificheUNET/groundTruth"

os.makedirs(output, exist_ok=True)


gruppiPref = defaultdict(list)

for nomeFile in sorted(os.listdir(input)):
    nomeCompleto = os.path.join(input, nomeFile)
    nomeNoEst = os.path.splitext(nomeFile)[0]

    match = re.match(r"(task-\d+-annotation-\d+-by-\d+-tag-[a-zA-Z0-9]+)-(\d+)", nomeNoEst) #controllo che sia un nome possibile

    if match:
        prefisso = match.group(1)
        gruppiPref[prefisso].append(nomeCompleto) #raggruppamento dei task per (task-\d+-annotation-\d+-by-\d+-tag-[a-zA-Z0-9]+)-(\d+)

maschere = {}

for prefisso, listaFile in gruppiPref.items():

    maschera = cv2.imread(listaFile[0], 0) #lettura maschera in scala di grigi
    
    for percorso in listaFile[1:]:
        img = cv2.imread(percorso, 0)
        maschera = (maschera > 0) | (img > 0) #creazione maschera incrementale con binarizzazione
        
    maschere[prefisso] = maschera #maschera finale 




gruppiTask = defaultdict(dict)

for prefisso, maschera in maschere.items():
    matchTask = re.search(r"(task-\d+)", prefisso) 
    
    matchTag = re.search(r"tag-([a-zA-Z0-9]+)", prefisso) 

    if matchTask and matchTag:
        idTask = matchTask.group(1)  
        nomeTag = matchTag.group(1)  
        gruppiTask[idTask][nomeTag] = maschera #mashera per task di mais o infestante

for idTask, dizionarioTag in gruppiTask.items():
    dim = next(iter(dizionarioTag.values())).shape

    B = np.zeros(dim, dtype=np.uint8)
    G = np.zeros(dim, dtype=np.uint8)
    R = np.zeros(dim, dtype=np.uint8)

    if "Mais" in dizionarioTag:
        G = 128*dizionarioTag["Mais"].astype(dtype='uint8') #assegnamento del colore verde al mais
    
        
    if "Weeds" in dizionarioTag:
        R = 128*dizionarioTag["Weeds"].astype(dtype='uint8') #assegnamento del colore rosso alle infestanti
        G[dizionarioTag["Weeds"] > 0] = 0 #per pixel annotati sia come mais che infestante, viene segnato come infestante
        
    merge = cv2.merge([B, G, R]) #merge tra mais e infestanti

    nomeFile = f"{idTask}.png"
    percorsoSalvataggio = os.path.join(output, nomeFile)
    cv2.imwrite(percorsoSalvataggio, merge)

