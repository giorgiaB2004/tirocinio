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

cd /data/giorgiabartoli/unet

PASSO=256
GRUOUNDTRUTHIN="/data/giorgiabartoli/unet/groundTruth"
GRUOUNDTRUTHOUT="/data/giorgiabartoli/unet/groundTruthTiles"
MASKSIN="/data/giorgiabartoli/unet/masks"
MASKSOUT="/data/giorgiabartoli/unet/masksTiles"


rm -rf /data/giorgiabartoli/unet/dataset/
rm -rf /data/giorgiabartoli/unet/groundTruthTiles/
rm -rf /data/giorgiabartoli/unet/masksTiles/
python /data/giorgiabartoli/unet/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO
python /data/giorgiabartoli/unet/step02_creazioneTiles.py --inputDir "$MASKSIN" --outputDir "$MASKSOUT" --passo $PASSO
python /data/giorgiabartoli/unet/step03_creazioneDataset.py


echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"

SEED=42
echo "=== INIZIO PIPELINE 1 (SEED 42): $(date) ==="
{
    rm -rf __pycache__/
    
    echo "=== INIZIO ADDESTRAMENTO ==="
    python train.py --seed $SEED
    echo "=== FINE ADDESTRAMENTO ==="

    echo "=== INIZIO TEST ==="
    python test.py --seed $SEED

    mv grafico.png "grafico_passo${PASSO}.png"
    mv results/ "results_passo${PASSO}"
    mv files/ "files_passo${PASSO}"
} > "log_passo${PASSO}.log" 2>&1  
echo "=== PIPELINE 1 COMPLETATA: $(date) ==="


PASSO=128
rm -rf /data/giorgiabartoli/unet/dataset/
rm -rf /data/giorgiabartoli/unet/groundTruthTiles/
rm -rf /data/giorgiabartoli/unet/masksTiles/
python /data/giorgiabartoli/unet/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO
python /data/giorgiabartoli/unet/step02_creazioneTiles.py --inputDir "$MASKSIN" --outputDir "$MASKSOUT" --passo $PASSO
python /data/giorgiabartoli/unet/step03_creazioneDataset.py

SEED=42
echo "=== INIZIO PIPELINE 4 (SEED 42): $(date) ==="
{
    rm -rf __pycache__/
    
    echo "=== INIZIO ADDESTRAMENTO ==="
    python train.py --seed $SEED
    echo "=== FINE ADDESTRAMENTO ==="

    echo "=== INIZIO TEST ==="
    python test.py --seed $SEED

    mv grafico.png "grafico_passo${PASSO}.png"
    mv results/ "results_passo${PASSO}"
    mv files/ "files_passo${PASSO}"
} > "log_passo${PASSO}.log" 2>&1   
echo "=== PIPELINE 4 COMPLETATA: $(date) ==="

PASSO=64


rm -rf /data/giorgiabartoli/unet/dataset/
rm -rf /data/giorgiabartoli/unet/groundTruthTiles/
rm -rf /data/giorgiabartoli/unet/masksTiles/
python /data/giorgiabartoli/unet/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO
python /data/giorgiabartoli/unet/step02_creazioneTiles.py --inputDir "$MASKSIN" --outputDir "$MASKSOUT" --passo $PASSO
python /data/giorgiabartoli/unet/step03_creazioneDataset.py

SEED=42
echo "=== INIZIO PIPELINE 7 (SEED 42): $(date) ==="
{
    rm -rf __pycache__/
    
    echo "=== INIZIO ADDESTRAMENTO ==="
    python train.py --seed $SEED
    echo "=== FINE ADDESTRAMENTO ==="

    echo "=== INIZIO TEST ==="
    python test.py --seed $SEED

    mv grafico.png "grafico_passo${PASSO}.png"
    mv results/ "results_passo${PASSO}"
    mv files/ "files_passo${PASSO}"
} > "log_passo${PASSO}.log" 2>&1  
echo "=== PIPELINE 7 COMPLETATA: $(date) ==="


echo "=== TUTTI I JOB TERMINATI CON SUCCESSO ==="