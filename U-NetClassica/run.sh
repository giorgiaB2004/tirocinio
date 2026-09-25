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

echo "=== VERIFICA GPU IN BATCH ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"

echo "=== INIZIO ADDESTRAMENTO: $(date) ==="
python train.py
echo "=== FINE ADDESTRAMENTO: $(date) ==="

echo "=== INIZIO TEST: $(date) ==="
python test.py
echo "=== PIPELINE COMPLETATA: $(date) ==="