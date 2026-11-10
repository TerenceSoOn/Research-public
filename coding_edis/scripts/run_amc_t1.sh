#!/bin/bash

# ==============================================================================
# EDIS Best-of-N Experiment Runner (with Oversampling)
# ==============================================================================

# 1. GPU Configuration
# --------------------
export CUDA_VISIBLE_DEVICES="2"

# Multi-GPU Mode (requires: pip install accelerate>=0.20)
USE_MULTI_GPU="false"

# 2. Experiment Parameters
# ------------------------
MODEL_PATH="/home/pterence6699/coding_edis/pretrained/Qwen2.5-math-1.5B"
DATASET_PATH="/home/pterence6699/coding_edis/math_data/amc23_1010.parquet"

# Oversampling parameters (Paper Section 5.1)
# k: candidates to retain after filtering (paper: k=8)
# m: oversampling multiplier (paper: m ∈ {1, 2, 4, 8, 16})
# Total candidates per question = m * k
K=8
M=8

# Generation batch size (how many sequences per forward pass, to avoid OOM)
GEN_BATCH_SIZE=4

# Question Num: How many questions to process (use -1 for all)
MAX_SAMPLES=100

# Generation hyperparameters
TEMPERATURE=1.0
TOP_P=1.0
MAX_NEW_TOKENS=2048

# 3. Output Configuration
# -----------------------
OUTPUT_DIR="/home/pterence6699/coding_edis/run_result"

# ==============================================================================

N_TOTAL=$((M * K))

echo "==========================================="
echo "Running Best-of-N Experiment (Oversampling)"
echo "==========================================="
echo "Time        : $(date +"%Y%m%d_%H%M%S")"
echo "Model       : $MODEL_PATH"
echo "Dataset     : $DATASET_PATH"
echo "k (retain)  : $K"
echo "m (oversamp): $M"
echo "N (total)   : $N_TOTAL  (m × k)"
echo "Batch Size  : $GEN_BATCH_SIZE"
echo "Samples     : $MAX_SAMPLES"
echo "Temperature : $TEMPERATURE"
echo "Device      : CUDA ($CUDA_VISIBLE_DEVICES)"
echo "Multi-GPU   : $USE_MULTI_GPU"
echo "Output Dir  : $OUTPUT_DIR"
echo "==========================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the python script
MULTI_GPU_FLAG=""
if [ "$USE_MULTI_GPU" = "true" ]; then
    MULTI_GPU_FLAG="--use-multi-gpu"
fi

python3 "$(dirname "$0")/best_of_n.py" \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --k "$K" \
    --m "$M" \
    --gen-batch-size "$GEN_BATCH_SIZE" \
    --max-samples "$MAX_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "cuda" \
    $MULTI_GPU_FLAG


EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "==========================================="
    echo "Experiment Success!"
    echo "Results saved to: $OUTPUT_DIR/{dataset}/jsonl/"
    echo "Filename: m${M}_k${K}_T${TEMPERATURE}_Timestamp.jsonl"
    echo "Summary saved to: $OUTPUT_DIR/{dataset}/summary/"
    echo "==========================================="
else
    echo "Experiment Failed with exit code $EXIT_CODE"
fi
