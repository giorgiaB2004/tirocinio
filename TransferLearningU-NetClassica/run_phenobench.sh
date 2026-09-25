#!/bin/bash
#SBATCH --job-name=weed_segmentation    
#SBATCH --partition=long               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:1                   
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --output=log_%j.log     # <-- Unico log principale di Slurm (%j mette il Job ID)

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
cd /data/giorgiabartoli/phenoBenchUnet/Multiclass-Segmentation-in-PyTorch-main

# DATASET
PASSO=256
TESTIMGIN="/data/giorgiabartoli/phenoBenchUnet/PhenoBench/test/images"
TESTIMGOUT="/data/giorgiabartoli/phenoBenchUnet/dataset/test/images"

TRAINIMGIN="/data/giorgiabartoli/phenoBenchUnet/PhenoBench/train/images"
TRAINIMGOUT="/data/giorgiabartoli/phenoBenchUnet/dataset/train/images"
TRAINMASKIN="/data/giorgiabartoli/phenoBenchUnet/PhenoBench/train/semantics"
TRAINMASKOUT="/data/giorgiabartoli/phenoBenchUnet/dataset/train/masks"

VALIMGIN="/data/giorgiabartoli/phenoBenchUnet/PhenoBench/val/images"
VALIMGOUT="/data/giorgiabartoli/phenoBenchUnet/dataset/valid/images"
VALMASKIN="/data/giorgiabartoli/phenoBenchUnet/PhenoBench/val/semantics"
VALMASKOUT="/data/giorgiabartoli/phenoBenchUnet/dataset/valid/masks"

: <<'COMMENTO'
echo "=== inizio creazione dataset ==="
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$TESTIMGIN" --outputDir "$TESTIMGOUT" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$TRAINIMGIN" --outputDir "$TRAINIMGOUT" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$TRAINMASKIN" --outputDir "$TRAINMASKOUT" --passo $PASSO --isBN
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$VALIMGIN" --outputDir "$VALIMGOUT" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$VALMASKIN" --outputDir "$VALMASKOUT" --passo $PASSO --isBN
echo "=== fine creazione dataset ==="
COMMENTO


echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"

# ==========================================
# ESECUZIONE 1 (SEED 42)
# ==========================================
SEED=42
echo "=== INIZIO: $(date) ==="
{
    
    echo "=== INIZIO ADDESTRAMENTO ==="
    python trainBN.py --seed $SEED
    echo "=== FINE ADDESTRAMENTO ==="

    echo "=== INIZIO TEST ==="
    python testBN.py --seed $SEED

    mv grafico.png "grafico_losses_phenobench.png"
    mv results/ "results_phenobench"

} > "log_training_phenobench.log" 2>&1  
echo "=== FINE: $(date) ===" 


