#!/bin/bash
#SBATCH --job-name=yolo26_weed_segmentation    
#SBATCH --partition=long               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:1                   
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --output=log_%j.log     

# 1. PULIZIA E CARICAMENTO MODULI
module purge
module load cuda 

# 2. STRATEGIA PER LO STUB ERROR E OTTIMIZZAZIONI ULTRALYTICS
unset FORCE_CUDA
export CUDA_CACHE_DISABLE=0
export ULTRALYTICS_CONFIG_DIR="" # Evita conflitti di permessi nella home per Ultralytics

# 3. ATTIVAZIONE CONDA SICURA
source /home/giorgiabartoli/miniconda3/etc/profile.d/conda.sh
conda activate yolo #(da creare l'ambiente per yolo)

# 4. SPOSTATI NELLA CARTELLA DEL PROGETTO YOLO
# Sostituisci con il percorso reale dove tieni i file .py e .yaml di YOLO26
cd /data/giorgiabartoli/YOLO
PASSO=256
GRUOUNDTRUTHIN="/data/giorgiabartoli/YOLO/groundTruth"
GRUOUNDTRUTHOUT="/data/giorgiabartoli/YOLO/groundTruthTiles"
RAWIN="/data/giorgiabartoli/YOLO/raw"
RAWOUT="/data/giorgiabartoli/YOLO/rawTiles"

echo "=== inizio creazione dataset ==="
python /data/giorgiabartoli/YOLO/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO --isBN
python /data/giorgiabartoli/YOLO/step02_creazioneTiles.py --inputDir "$RAWIN" --outputDir "$RAWOUT" --passo $PASSO
python /data/giorgiabartoli/YOLO/step03_creazioneDataset.py
echo "=== fine creazione dataset ==="

echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available()); print('Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nessuno')"

# ==========================================
# ESECUZIONE FLUSSO YOLO26
# ==========================================
echo "=== INIZIO PIPELINE YOLO26: $(date) ==="

#echo "=== FASE 1: INIZIO PRE-ADDESTRAMENTO (PhenoBench) ==="
#python pretrain.py
#echo "=== FASE 1: COMPLETATA ==="

echo "----------------------------------------------------"

echo "=== FASE 2: INIZIO FINE-TUNING (I tuoi dati 256x256) ==="
python train.py
echo "=== FASE 2: COMPLETATA ==="

echo "----------------------------------------------------"

echo "=== FASE 3: INIZIO TEST FINALE ==="
python predict.py
echo "=== FASE 3: COMPLETATA ==="


echo "=== FINE PIPELINE YOLO26: $(date) ==="