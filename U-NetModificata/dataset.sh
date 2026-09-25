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

module purge
module load cuda 

unset FORCE_CUDA
export CUDA_CACHE_DISABLE=0

source /home/giorgiabartoli/miniconda3/etc/profile.d/conda.sh
conda activate unet_env

cd /data/giorgiabartoli/modificheUNET



PASSO=64
GRUOUNDTRUTHIN="/data/giorgiabartoli/modificheUNET/groundTruth"
GRUOUNDTRUTHOUT="/data/giorgiabartoli/modificheUNET/groundTruthTiles64"
RAWIN="/data/giorgiabartoli/modificheUNET/raw"
RAWOUT="/data/giorgiabartoli/modificheUNET/rawTiles64"

echo "=== inizio creazione dataset ==="
python /data/giorgiabartoli/modificheUNET/step02_creazioneTiles.py --inputDir "$GRUOUNDTRUTHIN" --outputDir "$GRUOUNDTRUTHOUT" --passo $PASSO
python /data/giorgiabartoli/modificheUNET/step02_creazioneTiles.py --inputDir "$RAWIN" --outputDir "$RAWOUT" --passo $PASSO
python /data/giorgiabartoli/modificheUNET/step03_creazioneDataset.py
echo "=== fine creazione dataset ==="