#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    echo "[Info] Re-executing with bash..."
    exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_EMPO_ROOT="${VERL_EMPO_ROOT:-$(cd "${PROJECT_ROOT}/../coding_empo/project_EMPO/EMPO/verl-empo" && pwd)}"
PROJECT_EMPO_ROOT="${PROJECT_EMPO_ROOT:-$(cd "${VERL_EMPO_ROOT}/../.." && pwd)}"
source "${SCRIPT_DIR}/lib/runtime_env.sh"
resolve_runtime_defaults
resolve_livecodebench_dataset_path

# =========================
# Offline Eval Config
# =========================
# Choose any subset of: math,nl,coding
# math   -> AMC23, AIME24, MATH500, GSM8K
# nl     -> MMLU-Pro, GPQA-Diamond
# coding -> LiveCodeBench
RUN_ID="eval_$(date +%Y%m%d_%H%M%S)"
GPU_IDS="6"
EVAL_GROUPS="math, nl, coding"
FAIL_FAST="false"

# Generation sample counts. For final reporting use 16 for math/NL and 10 for LCB.
MATH_N=1
NL_N=1
LIVECODEBENCH_N=1

# Sampling config. Keep math/NL and LiveCodeBench explicit so eval settings are reproducible.
VAL_DO_SAMPLE=True
VAL_TEMPERATURE=0.0
VAL_TOP_P=0.95
VAL_TOP_K=20
LIVECODEBENCH_TEMPERATURE=0.0
LIVECODEBENCH_TOP_P=0.95

# Keep dataset subsets fixed, but randomize generation by default so repeated
# eval runs behave like online validation. Set EVAL_SEED to an integer to
# reproduce a run exactly.
SUBSET_SEED=20260418
EVAL_SEED="${EVAL_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}"

MMLU_SUBSET_SIZE=100
MATH500_SUBSET_SIZE=100
GSM8K_SUBSET_SIZE=100
GPQA_SUBSET_SIZE=100

# Smoke controls. Use empty values for full evaluation.
MATH_ROW_LIMIT=""
NL_ROW_LIMIT=""

# LiveCodeBench config. For coding-only eval on one clean GPU, set:
# GPU_IDS="6" and LIVECODEBENCH_TENSOR_PARALLEL_SIZE=1
LIVECODEBENCH_RELEASE="release_v6"
LIVECODEBENCH_SUBSET_SIZE=100
LIVECODEBENCH_TENSOR_PARALLEL_SIZE=2
LIVECODEBENCH_NUM_PROCESS_EVALUATE=12
LIVECODEBENCH_TIMEOUT=6
LIVECODEBENCH_VLLM_USE_V1="0"

# Optional memory/runtime controls.
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=3072
VAL_BATCH_SIZE=null
GPU_MEMORY_UTILIZATION=0.8

# Checkpoints to evaluate. Format:
#   CHECKPOINTS+=("id|/absolute/path/to/checkpoint")
CHECKPOINTS=()
CHECKPOINTS+=("qwen2.5_A_1500_5297|/home/pterence6699/coding_edis_empo/outputs/outptus-qwen2.5-1.5b-math-2/A_oversampled-off_weighted-off/checkpoints/global_step_1500")
CHECKPOINTS+=("qwen2.5_B_1440_5453|/home/pterence6699/coding_edis_empo/outputs/outptus-qwen2.5-1.5b-math-2/B_oversampled-on_weighted-off/checkpoints/top_amc_checkpoints/global_step_1440_score_0.545312")
CHECKPOINTS+=("qwen2.5_C_1000_5172|/home/pterence6699/coding_edis_empo/outputs/outptus-qwen2.5-1.5b-math-2/C_oversampled-off_weighted-on/checkpoints/global_step_1000")
CHECKPOINTS+=("qwen2.5_D_1530_5688|/home/pterence6699/coding_edis_empo/outputs/outptus-qwen2.5-1.5b-math-2/D_oversampled-on_weighted-on/checkpoints/top_amc_checkpoints/global_step_1530_score_0.568750")
CHECKPOINTS+=("qwen2.5-baseline|/home/pterence6699/coding_edis_empo/runtime/pretrained/Qwen2.5-math-1.5B")


# =========================
# Runtime Setup
# =========================
VENV_PATH="${TRAIN_VENV:-}"
if [ -z "${VENV_PATH}" ]; then
    resolve_train_venv
else
    if [ ! -x "${VENV_PATH}/bin/python3" ]; then
        echo "[Error] TRAIN_VENV=${VENV_PATH} does not contain bin/python3." >&2
        exit 1
    fi
fi

if [ ! -x "${VENV_PATH}/bin/python3" ]; then
    echo "[Error] No usable training Python venv found." >&2
    echo "        Set TRAIN_VENV=/path/to/venv and retry." >&2
    exit 1
fi

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "[Error] CHECKPOINTS is empty. Add at least one checkpoint in this script." >&2
    exit 1
fi

if [ "${GPU_IDS}" = "auto" ]; then
    echo "[Error] GPU_IDS must be set manually in this script, for example GPU_IDS=\"4,5\"." >&2
    exit 1
fi

if ! [[ "${SUBSET_SEED}" =~ ^[0-9]+$ ]]; then
    echo "[Error] SUBSET_SEED must be a non-negative integer, got ${SUBSET_SEED}." >&2
    exit 1
fi
if ! [[ "${EVAL_SEED}" =~ ^[0-9]+$ ]]; then
    echo "[Error] EVAL_SEED must be a non-negative integer, got ${EVAL_SEED}." >&2
    exit 1
fi

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${VERL_EMPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

QUEUE_FILE="${PROJECT_ROOT}/eval_runs/${RUN_ID}/queue.generated.yaml"
mkdir -p "$(dirname "${QUEUE_FILE}")"

{
    printf 'run_id: %s\n' "${RUN_ID}"
    printf 'output_root: %s\n' "${PROJECT_ROOT}/eval_runs"
    printf 'gpu_ids: "%s"\n' "${GPU_IDS}"
    printf 'fail_fast: %s\n' "${FAIL_FAST}"
    printf 'runtime_root: %s\n' "${RUNTIME_ROOT}"
    printf 'data_root: %s\n' "${DATA_ROOT}"
    printf 'model_path: %s\n' "${MODEL_PATH}"
    printf 'train_venv: %s\n' "${VENV_PATH}"
    printf 'subset_seed: %s\n' "${SUBSET_SEED}"
    printf 'eval_seed: %s\n' "${EVAL_SEED}"
    printf 'groups:\n'
    IFS=',' read -ra GROUP_ARRAY <<< "${EVAL_GROUPS}"
    for group in "${GROUP_ARRAY[@]}"; do
        group="${group//[[:space:]]/}"
        case "${group}" in
            math|nl|coding)
                printf '  - %s\n' "${group}"
                ;;
            "")
                ;;
            *)
                echo "[Error] Unsupported group '${group}'. Use math,nl,coding." >&2
                exit 1
                ;;
        esac
    done
    printf 'math_n: %s\n' "${MATH_N}"
    printf 'nl_n: %s\n' "${NL_N}"
    printf 'val_do_sample: %s\n' "${VAL_DO_SAMPLE}"
    printf 'val_temperature: %s\n' "${VAL_TEMPERATURE}"
    printf 'val_top_p: %s\n' "${VAL_TOP_P}"
    printf 'val_top_k: %s\n' "${VAL_TOP_K}"
    printf 'mmlu_subset_size: %s\n' "${MMLU_SUBSET_SIZE}"
    printf 'math500_subset_size: %s\n' "${MATH500_SUBSET_SIZE}"
    printf 'gsm8k_subset_size: %s\n' "${GSM8K_SUBSET_SIZE}"
    printf 'gpqa_subset_size: %s\n' "${GPQA_SUBSET_SIZE}"
    printf 'livecodebench_n: %s\n' "${LIVECODEBENCH_N}"
    printf 'livecodebench_temperature: %s\n' "${LIVECODEBENCH_TEMPERATURE}"
    printf 'livecodebench_top_p: %s\n' "${LIVECODEBENCH_TOP_P}"
    printf 'livecodebench_release: %s\n' "${LIVECODEBENCH_RELEASE}"
    printf 'livecodebench_subset_size: %s\n' "${LIVECODEBENCH_SUBSET_SIZE}"
    printf 'livecodebench_local_dataset_path: %s\n' "${LIVECODEBENCH_LOCAL_DATASET_PATH}"
    printf 'livecodebench_tensor_parallel_size: %s\n' "${LIVECODEBENCH_TENSOR_PARALLEL_SIZE}"
    printf 'livecodebench_num_process_evaluate: %s\n' "${LIVECODEBENCH_NUM_PROCESS_EVALUATE}"
    printf 'livecodebench_timeout: %s\n' "${LIVECODEBENCH_TIMEOUT}"
    printf 'livecodebench_vllm_use_v1: "%s"\n' "${LIVECODEBENCH_VLLM_USE_V1}"
    printf 'max_prompt_length: %s\n' "${MAX_PROMPT_LENGTH}"
    printf 'max_response_length: %s\n' "${MAX_RESPONSE_LENGTH}"
    printf 'val_batch_size: %s\n' "${VAL_BATCH_SIZE}"
    printf 'gpu_memory_utilization: %s\n' "${GPU_MEMORY_UTILIZATION}"
    if [ -n "${MATH_ROW_LIMIT}" ]; then
        printf 'math_row_limit: %s\n' "${MATH_ROW_LIMIT}"
    else
        printf 'math_row_limit: null\n'
    fi
    if [ -n "${NL_ROW_LIMIT}" ]; then
        printf 'nl_row_limit: %s\n' "${NL_ROW_LIMIT}"
    else
        printf 'nl_row_limit: null\n'
    fi
    printf 'checkpoints:\n'
    for item in "${CHECKPOINTS[@]}"; do
        checkpoint_id="${item%%|*}"
        checkpoint_path="${item#*|}"
        printf '  - id: %s\n' "${checkpoint_id}"
        printf '    path: %s\n' "${checkpoint_path}"
    done
} > "${QUEUE_FILE}"

cd "${PROJECT_ROOT}"
echo "[OfflineEval] Generated config: ${QUEUE_FILE}"
python -m evaluation.run_offline_eval_queue "${QUEUE_FILE}"

SUMMARY_SIMPLE="${PROJECT_ROOT}/eval_runs/${RUN_ID}/summary_simple.txt"
SUMMARY_JSON="${PROJECT_ROOT}/eval_runs/${RUN_ID}/summary_simple.json"
echo
echo "[OfflineEval] Simple result: ${SUMMARY_SIMPLE}"
cat "${SUMMARY_SIMPLE}"
echo
echo "[OfflineEval] Simple JSON: ${SUMMARY_JSON}"
