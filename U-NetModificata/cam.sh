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


python layerCAM.py \
    --test_dir /data/giorgiabartoli/modificheUNET/maisweed256/test/images \
    --output_dir ./layercam_results256 \
    --checkpoint_path /data/giorgiabartoli/modificheUNET/passo256/noSigReg_gc/noSigReg_gc/files/checkpoint_run_none_aspp_drop0.3_ch_green_chromaticity_reg0_seed43.pth
: <<'COMMENTO'
python layerCAM.py \
    --test_dir /data/giorgiabartoli/modificheUNET/maisweed128/test/images \
    --output_dir ./layercam_results128 \
    --checkpoint_path /data/giorgiabartoli/modificheUNET/passo128/exg/files/checkpoint_run_dual_aspp_drop0.3_ch_exg_reg1e-3_seed43.pth

python layerCAM.py \
    --test_dir /data/giorgiabartoli/modificheUNET/maisweed64/test/images \
    --output_dir ./layercam_results64 \
    --checkpoint_path /data/giorgiabartoli/modificheUNET/passo64/exg/files/checkpoint_run_dual_noaspp_drop0.0_ch_exg_reg1e-3_seed43.pth

COMMENTO    

python layerCAM.py \
    --test_dir /data/giorgiabartoli/modificheUNET/maisweed256/test/images \
    --output_dir ./layercam_results256_aspp_project_0\
    --checkpoint_path /data/giorgiabartoli/modificheUNET/passo256/noSigReg_gc/noSigReg_gc/files/checkpoint_run_none_aspp_drop0.3_ch_green_chromaticity_reg0_seed43.pth \
    --target_layer aspp.project.0


python layerCAM.py \
    --test_dir /data/giorgiabartoli/modificheUNET/maisweed256/test/images \
    --output_dir ./layercam_results256_e1_conv_conv_3 \
    --checkpoint_path /data/giorgiabartoli/modificheUNET/passo256/noSigReg_gc/noSigReg_gc/files/checkpoint_run_none_aspp_drop0.3_ch_green_chromaticity_reg0_seed43.pth \
    --target_layer e1.conv.conv.3
