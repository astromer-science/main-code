#!/bin/bash

# ==========================================
# 1. DEFAULT CONFIGURATION
# (Used if no arguments are provided)
# ==========================================
GPU_ID="0"
BATCH_SIZE=2500
BASE_MODEL_PATH="" # Empty by default (means no base model comparison)

# Default lists (space separated strings for parsing later)
DATASETS="alcock"
FOLDS="0 1 2"
SPCS="20"

# Base directories
ROOT_RESULTS="./presentation/results/v2/finetuning"
ROOT_DATA="./data/records"

# ==========================================
# 2. ARGUMENT PARSING
# ==========================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --bs)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --base_model)
      BASE_MODEL_PATH="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --spcs)
      SPCS="$2"
      shift 2
      ;;
    --root_results)
      ROOT_RESULTS="$2"
      shift 2
      ;;
    --root_data)
      ROOT_DATA="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Convert string inputs to Arrays
IFS=' ' read -r -a DATASETS_ARR <<< "$DATASETS"
IFS=' ' read -r -a FOLDS_ARR <<< "$FOLDS"
IFS=' ' read -r -a SPCS_ARR <<< "$SPCS"

# ==========================================
# 3. DEBUG INFO
# ==========================================
echo "Configuration:"
echo "  GPU: $GPU_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  Datasets: ${DATASETS_ARR[*]}"
echo "  Folds: ${FOLDS_ARR[*]}"
echo "  SPC: ${SPCS_ARR[*]}"
if [ -n "$BASE_MODEL_PATH" ]; then
    echo "  Base Model: $BASE_MODEL_PATH"
else
    echo "  Base Model: None (Skipping comparison)"
fi
echo "=========================================="

# ==========================================
# 4. MAIN LOOP
# ==========================================

for dataset in "${DATASETS_ARR[@]}"; do
    echo "Processing Dataset: $dataset"

    for fold in "${FOLDS_ARR[@]}"; do
        
        for spc in "${SPCS_ARR[@]}"; do
            
            # Construct dynamic paths
            # Model path: ./presentation/results/v2/finetuning/alcock/fold_0/alcock_20
            MODEL_PATH="${ROOT_RESULTS}/${dataset}/fold_${fold}/${dataset}_${spc}"
            
            # Data path: ./data/records/alcock/fold_0/alcock_20
            DATA_PATH="${ROOT_DATA}/${dataset}/fold_${fold}/${dataset}_${spc}"

            # Check if model directory exists
            if [ -d "$MODEL_PATH" ]; then
                echo "------------------------------------------------"
                echo "Running: Fold $fold | SPC $spc"
                
                # Construct the command using an array (safest way in bash)
                CMD=(python -m presentation.pipelines.pipeline_0.testing)
                CMD+=(--model "$MODEL_PATH")
                CMD+=(--dataset "$DATA_PATH")
                CMD+=(--gpu "$GPU_ID")
                CMD+=(--bs "$BATCH_SIZE")
                
                # Add base_model argument only if the variable is set
                if [ -n "$BASE_MODEL_PATH" ]; then
                    CMD+=(--base_model "$BASE_MODEL_PATH")
                fi
                
                # Execute the command
                "${CMD[@]}"
                
            else
                echo "[WARNING] Skipping: Model not found at $MODEL_PATH"
            fi

        done
    done
done

echo "========================================================"
echo " Execution Completed."
echo "========================================================"