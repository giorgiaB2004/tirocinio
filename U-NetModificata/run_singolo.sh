#!/bin/bash
#SBATCH --job-name=weed_segmentation    
#SBATCH --partition=long               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:1                   
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
# Unico log principale di Slurm (%j mette il Job ID)
#SBATCH --output=log_%j.log     

# 1. PULIZIA E CARICAMENTO MODULI
module purge
module load cuda 

# 2. STRATEGIA PER LO STUB ERROR
unset FORCE_CUDA
export CUDA_CACHE_DISABLE=0

# 3. ATTIVAZIONE CONDA SICURA
source /home/giorgiabartoli/miniconda3/etc/profile.d/conda.sh
conda activate unet_env

# 4. SPOSTATI NELLA CARTELLA DEL PROGETTO
cd /data/giorgiabartoli/modificheUNET




# DATASET
PASSO=256
GRUOUNDTRUTHIN="/data/giorgiabartoli/modificheUNET/groundTruth"
GRUOUNDTRUTHOUT="/data/giorgiabartoli/modificheUNET/groundTruthTiles"
RAWIN="/data/giorgiabartoli/modificheUNET/raw"
RAWOUT="/data/giorgiabartoli/modificheUNET/rawTiles"

echo "=== inizio creazione dataset ==="
python /data/giorgiabartoli/modificheUNET/step01_creazioneMaschere.py
python /data/giorgiabartoli/modificheUNET/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO
python /data/giorgiabartoli/modificheUNET/step02_creazioneTiles.py --inputDir "$RAWIN" --outputDir "$RAWOUT" --passo $PASSO
python /data/giorgiabartoli/modificheUNET/step03_creazioneDataset.py
echo "=== fine creazione dataset ==="
: <<'COMMENTO'
ATTENTIONS=("none" "class_aware" "dual" "both")
DROPOUTS=("0.0" "0.3")
ASPP_OPTIONS=("noaspp" "aspp")



echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"


SEED=43
att="none"
drop=0.0

echo "=== INIZIO: $(date) ==="

echo "=== INIZIO ADDESTRAMENTO ==="
python train.py --seed $SEED --attention_type "$att" --dropout_rate "$drop" 
echo "=== FINE ADDESTRAMENTO ==="

echo "=== INIZIO TEST ==="
python test.py --seed $SEED --attention_type "$att" --dropout_rate "$drop" 
            
COMMENTO
echo "=== FINE: $(date) ==="