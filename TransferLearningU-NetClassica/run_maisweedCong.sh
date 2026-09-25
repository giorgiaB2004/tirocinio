#!/bin/bash
#SBATCH --job-name=weed_segmentation    
#SBATCH --partition=long               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:1                   
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --output=log_%j.log    

module purge
module load cuda 

unset FORCE_CUDA
export CUDA_CACHE_DISABLE=0

source /home/giorgiabartoli/miniconda3/etc/profile.d/conda.sh
conda activate unet_env


cd /data/giorgiabartoli/phenoBenchUnet/Multiclass-Segmentation-in-PyTorch-main


: <<'COMMENTO'
# DATASET
PASSO=256
GRUOUNDTRUTHIN="/data/giorgiabartoli/phenoBenchUnet/groundTruth"
GRUOUNDTRUTHOUT="/data/giorgiabartoli/phenoBenchUnet/groundTruthTiles"
MASKSIN="/data/giorgiabartoli/phenoBenchUnet/masks"
MASKSOUT="/data/giorgiabartoli/phenoBenchUnet/masksTiles"

echo "=== inizio creazione dataset ==="
python /data/giorgiabartoli/phenoBenchUnet/step01_creazioneMaschere.py
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "$MASKSIN" --outputDir "$MASKSOUT" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step03_creazioneDataset.py
echo "=== fine creazione dataset ==="
COMMENTO
echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"


SEED=42
echo "=== INIZIO: $(date) ==="
{    
    echo "=== INIZIO ADDESTRAMENTO ==="
    python trainCongelamento.py --seed $SEED
    echo "=== FINE ADDESTRAMENTO ==="

    echo "=== INIZIO TEST ==="
    python test.py --seed $SEED

    mv grafico.png "prova.png"
    mv results/ "resultsprova"

} 
echo "=== FINE: $(date) ==="
