#!/usr/bin/env bash
# Pre-compute reference density targets using the split pipeline:
#   Step 1: Generate responses + extract answers (GPU, vLLM)
#   Step 2: Cluster extracted answers (CPU, multicore)
#   Step 3: Merge ref_density into training parquet
#
# Usage:
#   bash scripts/precompute_ref_density_pipeline.sh
#   MODEL_PATH=/path/to/model N_RESPONSES=32 bash scripts/precompute_ref_density_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Paths ---
VERL_EMPO_ROOT="${VERL_EMPO_ROOT:-$(cd "${PROJECT_ROOT}/../verl-empo" && pwd)}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/runtime/pretrained/qwen3-4b-base}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/runtime/math_data/NM_20K_1010.jsonl}"
PARQUET_PATH="${PARQUET_PATH:-${PROJECT_ROOT}/runtime/math_data/NM_20K_1010.parquet}"

# --- Generation parameters (match training defaults) ---
N_RESPONSES="${N_RESPONSES:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DENSITY_THRESHOLD="${DENSITY_THRESHOLD:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL_IMPL="${MODEL_IMPL:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"

# --- Clustering parameters ---
NUM_WORKERS="${NUM_WORKERS:-64}"

# --- Prompt limit (for testing) ---
MAX_PROMPTS="${MAX_PROMPTS:-}"

# --- Derived intermediate paths ---
DATA_STEM="$(basename "${DATA_PATH}" .jsonl)"
DATA_DIR="$(dirname "${DATA_PATH}")"
EXTRACTED_PATH="${EXTRACTED_PATH:-${DATA_DIR}/${DATA_STEM}.extracted_answers.jsonl}"
REF_DENSITY_PATH="${REF_DENSITY_PATH:-${DATA_DIR}/${DATA_STEM}.ref_density.jsonl}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${DATA_DIR}/${DATA_STEM}.ref_density.parquet}"

# --- Environment ---
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
export PYTHONPATH="${PROJECT_ROOT}:${VERL_EMPO_ROOT}:${PYTHONPATH:-}"
source "${VENV_PATH}/bin/activate"

echo "============================================================"
echo "Pre-compute Reference Density (Split Pipeline)"
echo "============================================================"
echo "  model:              ${MODEL_PATH}"
echo "  data:               ${DATA_PATH}"
echo "  parquet:            ${PARQUET_PATH}"
echo "  n_responses:        ${N_RESPONSES}"
echo "  temperature:        ${TEMPERATURE}"
echo "  top_p:              ${TOP_P}"
echo "  density_threshold:  ${DENSITY_THRESHOLD}"
echo "  model_impl:         ${MODEL_IMPL}"
echo "  batch_size:         ${BATCH_SIZE}"
echo "  num_workers:        ${NUM_WORKERS}"
echo "  seed:               ${SEED}"
echo "  GPUs:               $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "------------------------------------------------------------"
echo "  step 1 output:      ${EXTRACTED_PATH}"
echo "  step 2 output:      ${REF_DENSITY_PATH}"
echo "  step 3 output:      ${OUTPUT_PARQUET}"
echo "============================================================"

# ---- Step 1: Generate + Extract (GPU) ----
echo ""
echo "[Step 1/3] Generating responses and extracting answers..."
python3 "${PROJECT_ROOT}/src/precompute_ref_density_generate.py" \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_path "${EXTRACTED_PATH}" \
    --n_responses "${N_RESPONSES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_response_length "${MAX_RESPONSE_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --model_impl "${MODEL_IMPL}" \
    ${TENSOR_PARALLEL_SIZE:+--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"} \
    ${MAX_PROMPTS:+--max_prompts "${MAX_PROMPTS}"}

echo "[Step 1/3] Done. Output: ${EXTRACTED_PATH}"

# ---- Step 2: Cluster (CPU) ----
echo ""
echo "[Step 2/3] Clustering extracted answers..."
python3 "${PROJECT_ROOT}/src/precompute_ref_density_cluster.py" \
    --input_path "${EXTRACTED_PATH}" \
    --output_path "${REF_DENSITY_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --density_threshold "${DENSITY_THRESHOLD}"

echo "[Step 2/3] Done. Output: ${REF_DENSITY_PATH}"

# ---- Step 3: Merge into parquet ----
echo ""
echo "[Step 3/3] Merging ref_density into parquet..."
python3 "${PROJECT_ROOT}/scripts/merge_ref_density_to_parquet.py" \
    --jsonl "${REF_DENSITY_PATH}" \
    --parquet "${PARQUET_PATH}" \
    --output "${OUTPUT_PARQUET}"

echo "[Step 3/3] Done. Output: ${OUTPUT_PARQUET}"

echo ""
echo "============================================================"
echo "Pipeline complete."
echo "  Final parquet: ${OUTPUT_PARQUET}"
echo "============================================================"
