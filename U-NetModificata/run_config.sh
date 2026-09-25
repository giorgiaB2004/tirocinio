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

cd /data/giorgiabartoli/modificheUNET

mkdir -p logs_esperimenti

echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"

SEED=42
echo "=== FINESTRA DI ESPERIMENTI INIZIATA: $(date) ==="

ATTENTIONS=("none" "class_aware" "dual" "both")
DROPOUTS=("0.0" "0.3")
ASPP_OPTIONS=("noaspp" "aspp")

for att in "${ATTENTIONS[@]}"; do
    for drop in "${DROPOUTS[@]}"; do
        for aspp in "${ASPP_OPTIONS[@]}"; do
            
            ASPP_FLAG=""
            if [ "$aspp" == "aspp" ]; then
                ASPP_FLAG="--use_aspp"
            fi
            
            LOG_FILE="logs_esperimenti/log_${att}_${aspp}_drop${drop}.log"
            
            echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
            echo "CONFIGURAZIONE CORRENTE:" | tee -a "$LOG_FILE"
            echo "Attenzione: $att | Dropout: $drop | ASPP: $aspp" | tee -a "$LOG_FILE"
            echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
            
            echo "=== INIZIO ADDESTRAMENTO: $(date) ===" | tee -a "$LOG_FILE"
            
            # Esegue train.py salvando l'output sia nel log specifico che nel log principale di Slurm (grazie a tee -a)
            python train.py --seed $SEED --attention_type "$att" --dropout_rate "$drop" --lambda_reg 0 $ASPP_FLAG 2>&1 | tee -a "$LOG_FILE"
            
            echo "=== FINE ADDESTRAMENTO ===" | tee -a "$LOG_FILE"
            echo "=== INIZIO TEST ===" | tee -a "$LOG_FILE"
            
            # Esegue test.py facendo la stessa cosa
            python test.py --seed $SEED --attention_type "$att" --dropout_rate "$drop" --lambda_reg 0 $ASPP_FLAG 2>&1 | tee -a "$LOG_FILE"
            
            echo "=== FINE TEST: $(date) ===" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            
        done
    done
done

echo "=== FINE DI TUTTE LE 16 COMBINAZIONI: $(date) ==="