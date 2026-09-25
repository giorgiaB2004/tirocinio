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

DATASET_PATH="/data/giorgiabartoli/modificheUNET/maisweed128"
CHECKPOINT_DIR="passo128/files"
RESULTS_DIR="passo128/results"
LOGS_DIR="passo128/logs_esperimenti"

mkdir -p "$LOGS_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== VERIFICA GPU INIZIALE ==="
python -c "import torch; print('CUDA Disponibile nel job:', torch.cuda.is_available())"

SEED=42
EPOCHS=150
LR="8e-4"

FOURTH_CHANNEL="green_chromaticity"  
LAMBDA_REG="1e-3"     

echo "=== FINESTRA DI ESPERIMENTI INIZIATA: $(date) ==="
echo "Configurazione Globale: Canale4=$FOURTH_CHANNEL | LambdaReg=$LAMBDA_REG | Epochs=$EPOCHS"

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
            
            RUN_NAME="run_${att}_${aspp}_drop${drop}_ch_${FOURTH_CHANNEL}_reg${LAMBDA_REG}_seed${SEED}"
            
            LOG_FILE="${LOGS_DIR}/log_${RUN_NAME}.log"
            CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_${RUN_NAME}.pth"
            
            TEST_SAVE_DIR="${RESULTS_DIR}/${RUN_NAME}/test"
            
            echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
            echo "CONFIGURAZIONE CORRENTE: $RUN_NAME" | tee -a "$LOG_FILE"
            echo "Attenzione: $att | Dropout: $drop | ASPP: $aspp" | tee -a "$LOG_FILE"
            echo "Salvataggio risultati in: ${RESULTS_DIR}/${RUN_NAME}" | tee -a "$LOG_FILE"
            echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
            
            echo "=== INIZIO ADDESTRAMENTO: $(date) ===" | tee -a "$LOG_FILE"
            
            python train.py \
                --dataset_path "$DATASET_PATH" \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --results_dir "$RESULTS_DIR" \
                --run_name "$RUN_NAME" \
                --seed $SEED \
                --epochs $EPOCHS \
                --learning_rate "$LR" \
                --attention_type "$att" \
                --dropout_rate "$drop" \
                $ASPP_FLAG \
                --fourth_channel "$FOURTH_CHANNEL" \
                --lambda_reg "$LAMBDA_REG" \
                --amp \
                --deterministic 2>&1 | tee -a "$LOG_FILE"
            
            echo "=== FINE ADDESTRAMENTO ===" | tee -a "$LOG_FILE"
            
            if [ -f "$CHECKPOINT_PATH" ]; then
                echo "=== INIZIO TEST ===" | tee -a "$LOG_FILE"
                
                python test.py \
                    --checkpoint "$CHECKPOINT_PATH" \
                    --dataset_path "$DATASET_PATH" \
                    --save_dir "$TEST_SAVE_DIR" \
                    --fourth_channel "$FOURTH_CHANNEL" \
                    --attention_type "$att" \
                    --dropout_rate "$drop" \
                    $ASPP_FLAG \
                    --amp \
                    --save_joint 2>&1 | tee -a "$LOG_FILE"
                
                echo "=== FINE TEST: $(date) ===" | tee -a "$LOG_FILE"
            else
                echo "=== [ATTENZIONE] Checkpoint non trovato in $CHECKPOINT_PATH. Salto il test. ===" | tee -a "$LOG_FILE"
            fi
            
            echo "" | tee -a "$LOG_FILE"
            
        done
    done
done

echo "=== FINE DI TUTTE LE 16 COMBINAZIONI: $(date) ==="