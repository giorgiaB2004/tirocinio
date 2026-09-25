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
cd /data/giorgiabartoli/phenoBenchUnet/transferlearning

: <<'COMMENTO'
# DATASET
PASSO=256
GRUOUNDTRUTHIN="/data/giorgiabartoli/phenoBenchUnet/groundTruth"
GRUOUNDTRUTHOUT="/data/giorgiabartoli/phenoBenchUnet/groundTruthTiles"
MASKSIN="/data/giorgiabartoli/phenoBenchUnet/masks"
MASKSOUT="/data/giorgiabartoli/phenoBenchUnet/masksTiles"

echo "=== inizio creazione dataset ==="
#python /data/giorgiabartoli/phenoBenchUnet/step01_creazioneMaschere.py
#python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "/data/giorgiabartoli/phenoBenchUnet/PhenoBench/test/images" --outputDir "/data/giorgiabartoli/phenoBenchUnet/dataset/test/images" --passo $PASSO
#python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "/data/giorgiabartoli/phenoBenchUnet/PhenoBench/train/images" --outputDir "/data/giorgiabartoli/phenoBenchUnet/dataset/train/images" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "/data/giorgiabartoli/phenoBenchUnet/PhenoBench/train/semantics" --outputDir "/data/giorgiabartoli/phenoBenchUnet/dataset/train/masks" --passo $PASSO --isBN
#python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "/data/giorgiabartoli/phenoBenchUnet/PhenoBench/val/images" --outputDir "/data/giorgiabartoli/phenoBenchUnet/dataset/valid/images" --passo $PASSO
python /data/giorgiabartoli/phenoBenchUnet/step02_creazioneTiles.py --inputDir "/data/giorgiabartoli/phenoBenchUnet/PhenoBench/val/semantics" --outputDir "/data/giorgiabartoli/phenoBenchUnet/dataset/valid/masks" --passo $PASSO --isBN

#python /data/giorgiabartoli/phenoBenchUnet/step03_creazioneDataset.py
echo "=== fine creazione dataset ==="

COMMENTO
echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"

echo "=== INIZIO: $(date) ==="

#python trainBN.py  --dataset_path "/data/giorgiabartoli/phenoBenchUnet/dataset" --results_dir "resultsPreAdd" --attention_type "none" --dropout_rate 0.0 --run_name phenobench_pretrain --amp

#python trainCongelamento.py --dataset_path /data/giorgiabartoli/phenoBenchUnet/maisweed --pretrained_path /data/giorgiabartoli/phenoBenchUnet/transferlearning/pretrained_files/pretrained_phenobench_pretrain.pth --run_name phenobench_congelamento_run --attention_type "class_aware" --dropout_rate 0.3 --use_aspp --checkpoint_dir "/data/giorgiabartoli/phenoBenchUnet/transferlearning/files"

python test.py --checkpoint "/data/giorgiabartoli/phenoBenchUnet/transferlearning/files/checkpoint_phenobench_congelamento_run.pth" --dataset_path "/data/giorgiabartoli/phenoBenchUnet/maisweed" --save_dir "/data/giorgiabartoli/phenoBenchUnet/transferlearning/tetsResults" --fourth_channel "exg" --attention_type "class_aware" --dropout_rate 0.3 --use_aspp --amp --save_joint

echo "=== FINE: $(date) ==="
