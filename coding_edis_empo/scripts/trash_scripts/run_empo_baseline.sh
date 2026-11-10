#!/usr/bin/env bash
set -x

# If invoked via sh (dash), re-exec with bash so array syntax works
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[Info] Re-executing with bash to support arrays..."
    exec /usr/bin/env bash "$0" "$@"
fi

# ============================================================================
# Directory Setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to verl-empo (EMPO repo)
VERL_EMPO_ROOT="${VERL_EMPO_ROOT:-$(cd "${PROJECT_ROOT}/../coding_empo/project_EMPO/EMPO/verl-empo" && pwd)}"
EMPO_ROOT="${EMPO_ROOT:-$(cd "${VERL_EMPO_ROOT}/.." && pwd)}"
PROJECT_EMPO_ROOT="${PROJECT_EMPO_ROOT:-$(cd "${VERL_EMPO_ROOT}/../.." && pwd)}"

cd "${VERL_EMPO_ROOT}"

# PYTHONPATH: verl-empo only (pure EMPO, no EDIS)
export PYTHONPATH="${VERL_EMPO_ROOT}:${PYTHONPATH:-}"

# ============================================================================
# Virtual Environment
# ============================================================================
VENV_PATH="${PROJECT_ROOT}/.venv"
if [ ! -x "${VENV_PATH}/bin/python3" ]; then
    for CANDIDATE in "${PROJECT_EMPO_ROOT}/venv" "/home/pterence6699/coding_empo/venv"; do
        if [ -x "${CANDIDATE}/bin/python3" ]; then
            VENV_PATH="${CANDIDATE}"
            break
        fi
    done
fi

if [ ! -x "${VENV_PATH}/bin/python3" ]; then
    echo "[Error] No usable virtual environment found." >&2
    echo "        Expected one of:" >&2
    echo "        - ${PROJECT_ROOT}/.venv" >&2
    echo "        - ${PROJECT_EMPO_ROOT}/venv" >&2
    echo "        - /home/pterence6699/coding_empo/venv" >&2
    exit 1
fi

PYTHON_BIN="${VENV_PATH}/bin/python3"
source "${VENV_PATH}/bin/activate"

# ============================================================================
# Environment Variables
# ============================================================================
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export VLLM_USE_V1=1
export EVAL_MATH_PATH="${EMPO_ROOT}/eval_math"

# ============================================================================
# GPU Topology
# ============================================================================
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
export NCCL_NVLS_ENABLE="${HAS_NVLINK:-0}"

# ============================================================================
# EMPO Baseline Parameters (all configurable via env vars)
# ============================================================================

# --- EMPO core ---
target="${REWARD_TYPE:-entropy}"
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}     # Standard EMPO: no oversampling
use_wandb="${USE_WANDB:-True}"
wandb_mode="${WANDB_MODE:-online}"

# --- Training batch sizes ---
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-4}

# --- Sequence lengths ---
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 1))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 3))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-1024}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-0.0}

# --- Loss ---
loss_mode=${LOSS_MODE:-"gspo"}
loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}

# --- PPO / GRPO ---
adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-3e-4}
clip_ratio_high=${CLIP_RATIO_HIGH:-4e-4}

# --- Filter groups ---
enable_filter_groups=${ENABLE_FILTER_GROUPS:-False}
confidence_lower_bound=${CONFIDENCE_LOWER_BOUND:-0.3}
confidence_upper_bound=${CONFIDENCE_UPPER_BOUND:-1.0}
filter_groups_metric=${FILTER_GROUPS_METRIC:-score}

# --- Optimizer ---
learning_rate=${LEARNING_RATE:-5e-7}
lr_warmup_steps=${LR_WARMUP_STEPS:-0}
weight_decay=${WEIGHT_DECAY:-0}

# --- Sampling ---
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}

# --- Validation sampling ---
val_do_sample=${VAL_DO_SAMPLE:-True}
val_n=${VAL_N:-16}
val_top_k=${VAL_TOP_K:-20}
val_top_p=${VAL_TOP_P:-0.95}
val_temperature=${VAL_TEMPERATURE:-0.2}

# --- Training schedule ---
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-300}
total_epochs=${TOTAL_EPOCHS:-1000}
total_training_steps=${TOTAL_TRAINING_STEPS:-1500}
val_before_train=${VAL_BEFORE_TRAIN:-True}
val_only=${VAL_ONLY:-False}

# --- Dynamic batching ---
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
infer_micro_batch_size=${INFER_MICRO_BATCH_SIZE:-null}
train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE:-null}
offload=${OFFLOAD:-True}

# --- GPU memory ---
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.6}

case "${use_wandb}" in
    True|true|1|yes|on)
        trainer_logger="['console','tensorboard','wandb']"
        ;;
    False|false|0|no|off)
        trainer_logger="['console','tensorboard']"
        ;;
    *)
        echo "[Error] Unsupported USE_WANDB=${use_wandb}. Use True/False (or on/off)." >&2
        exit 1
        ;;
esac

# ============================================================================
# Experiment Naming
# ============================================================================
project_name=${PROJECT_NAME:-'EMPO-Baseline'}
run_signature="baseline_N${n_resp_per_prompt}_rw-empo_${target}"
exp_name=${EXP_NAME:-"EMPO-Qwen2.5-1.5B-${run_signature}"}

# ============================================================================
# Ray
# ============================================================================
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
WORLD_SIZE=$((N_GPUS_PER_NODE * NNODES))
export RAY_DISABLE_DASHBOARD=${RAY_DISABLE_DASHBOARD:-1}

# ============================================================================
# Model & Data Paths
# ============================================================================
RAY_DATA_HOME=${RAY_DATA_HOME:-"${PROJECT_EMPO_ROOT}/ray_data"}
MODEL_PATH=${MODEL_PATH:-"/home/pterence6699/coding_edis/pretrained/Qwen2.5-math-1.5B"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${run_signature}"}
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_DIR}/checkpoints"}
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard_log"
export WANDB_DIR="${PROJECT_EMPO_ROOT}/wandb"
export WANDB_MODE="${wandb_mode}"

mkdir -p "${CKPTS_DIR}" "${OUTPUT_DIR}/console_logs" "${TENSORBOARD_DIR}"
echo "[Info] Output dir:  ${OUTPUT_DIR}"

# ============================================================================
# Resume
# ============================================================================
RESUME_STEP=${RESUME_STEP:-}
RESUME_PATH=${RESUME_PATH:-}

if [ -z "${RESUME_PATH}" ] && [ -n "${RESUME_STEP}" ]; then
    RESUME_PATH="${CKPTS_DIR}/global_step_${RESUME_STEP}"
fi

RESUME_ARGS=()
if [ -n "${RESUME_PATH}" ]; then
    if [ ! -d "${RESUME_PATH}" ]; then
        echo "[Error] Resume checkpoint directory ${RESUME_PATH} not found."
        exit 1
    fi

    ACTOR_DIR="${RESUME_PATH}/actor"
    if [ ! -d "${ACTOR_DIR}" ]; then
        echo "[Error] ${RESUME_PATH} does not contain an actor checkpoint directory."
        exit 1
    fi

    EXPECTED_SHARD="${ACTOR_DIR}/model_world_size_${WORLD_SIZE}_rank_0.pt"
    if [ ! -f "${EXPECTED_SHARD}" ]; then
        echo "[Error] ${EXPECTED_SHARD} not found. The checkpoint appears to be saved with a different world size." >&2
        exit 1
    fi

    RESUME_ARGS+=("trainer.resume_from_path=${RESUME_PATH}")
    RESUME_ARGS+=("trainer.resume_mode=resume_path")
fi

# ============================================================================
# Data Paths (configurable, multi-dataset val support)
# ============================================================================
test_data_dir="${EMPO_ROOT}/math_data"

# Training data
train_path="${TRAIN_DATA:-${EMPO_ROOT}/math_data/NM_20K_1010.parquet}"

# Validation data
amc23="${test_data_dir}/amc23_1010.parquet"
math500="${test_data_dir}/math500_1010.parquet"
aime24="${test_data_dir}/aime24_1010.parquet"

# Default: amc23. Override with VAL_FILES env var (comma-separated)
if [ -n "${VAL_FILES:-}" ]; then
    IFS=',' read -ra VAL_ARRAY <<< "${VAL_FILES}"
    TEST_FILES_STR=""
    for vf in "${VAL_ARRAY[@]}"; do
        vf=$(echo "$vf" | xargs)
        case "$vf" in
            amc23)   vf="${amc23}" ;;
            math500) vf="${math500}" ;;
            aime24)  vf="${aime24}" ;;
        esac
        if [ -n "${TEST_FILES_STR}" ]; then
            TEST_FILES_STR="${TEST_FILES_STR},'${vf}'"
        else
            TEST_FILES_STR="'${vf}'"
        fi
    done
    TEST_FILES="[${TEST_FILES_STR}]"
else
    TEST_FILES="['$amc23']"
fi

TRAIN_FILES="['$train_path']"

echo "===== EMPO Baseline Configuration ====="
echo "  N (responses):     ${n_resp_per_prompt}"
echo "  reward_type:       ${target}"
echo "  trainer_logger:    ${trainer_logger}"
echo "  use_wandb:         ${use_wandb}"
echo "  wandb_mode:        ${wandb_mode}"
echo "  temperature:       ${temperature}"
echo "  learning_rate:     ${learning_rate}"
echo "  model:             ${MODEL_PATH}"
echo "  train_data:        ${train_path}"
echo "  val_data:          ${TEST_FILES}"
echo "  output_dir:        ${OUTPUT_DIR}"
echo "========================================"

# ============================================================================
# Hydra Arguments
# ============================================================================
HYDRA_ARGS=(
    "++data.train_files=${TRAIN_FILES}"
    "++data.val_files=${TEST_FILES}"
    "++data.use_shm=False"
    "++data.prompt_key=prompt"
    "++data.truncation='left'"
    "++data.max_prompt_length=${max_prompt_length}"
    "++data.max_response_length=${max_response_length}"
    "++data.gen_batch_size=${gen_prompt_bsz}"
    "++data.train_batch_size=${train_prompt_bsz}"
    "++data.filter_overlong_prompts=False"
    "++actor_rollout_ref.rollout.n=${n_resp_per_prompt}"
    "++actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}"
    "++actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}"
    "++actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}"
    "++actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}"
    "++actor_rollout_ref.actor.clip_ratio_c=10.0"
    "++actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}"
    "++algorithm.adv_estimator=${adv_estimator}"
    "++algorithm.use_kl_in_reward=${use_kl_in_reward}"
    "++algorithm.kl_ctrl.kl_coef=${kl_coef}"
    "++algorithm.filter_groups.enable=${enable_filter_groups}"
    "++algorithm.filter_groups.metric=${filter_groups_metric}"
    "++algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}"
    "++algorithm.filter_groups.confidence_lower_bound=${confidence_lower_bound}"
    "++algorithm.filter_groups.confidence_upper_bound=${confidence_upper_bound}"
    "++actor_rollout_ref.model.use_remove_padding=True"
    "++actor_rollout_ref.model.use_fused_kernels=False"
    "++actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}"
    "++actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))"
    "++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))"
    "++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))"
    "++actor_rollout_ref.model.path=${MODEL_PATH}"
    "++actor_rollout_ref.model.override_config.attention_dropout=0."
    "++actor_rollout_ref.model.override_config.embd_pdrop=0."
    "++actor_rollout_ref.model.override_config.resid_pdrop=0."
    "++actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "++actor_rollout_ref.model.use_liger=False"
    "++actor_rollout_ref.actor.optim.lr=${learning_rate}"
    "++actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps}"
    "++actor_rollout_ref.actor.optim.weight_decay=${weight_decay}"
    "++actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}"
    "++actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size}"
    "++actor_rollout_ref.actor.fsdp_config.param_offload=${offload}"
    "++actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}"
    "++actor_rollout_ref.actor.entropy_coeff=0"
    "++actor_rollout_ref.actor.grad_clip=1.0"
    "++actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}"
    "++actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
    "++actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
    "++actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size}"
    "++actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "++actor_rollout_ref.rollout.expert_parallel=False"
    "++actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "++actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
    "++actor_rollout_ref.rollout.temperature=${temperature}"
    "++actor_rollout_ref.rollout.top_p=${top_p}"
    "++actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}"
    "++actor_rollout_ref.rollout.val_kwargs.n=${val_n}"
    "++actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}"
    "++actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
    "++actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}"
    "++actor_rollout_ref.rollout.enforce_eager=False"
    "++actor_rollout_ref.rollout.free_cache_engine=False"
    "++actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size}"
    "++actor_rollout_ref.ref.fsdp_config.param_offload=${offload}"
    "++actor_rollout_ref.ref.ulysses_sequence_parallel_size=1"
    "++actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
    # Pure EMPO reward config
    "++reward_model.reward_manager=empo"
    "++reward_model.reward_kwargs.reward_type=${target}"
    "++reward_model.overlong_buffer.enable=${enable_overlong_buffer}"
    "++reward_model.overlong_buffer.len=${overlong_buffer_len}"
    "++reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}"
    "++reward_model.use_xverify=False"
    "++reward_model.use_eval_math=True"
    "++trainer.logger=${trainer_logger}"
    "++trainer.project_name=${project_name}"
    "++trainer.experiment_name=${exp_name}"
    "++trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "++trainer.nnodes=${NNODES}"
    "++trainer.val_before_train=${val_before_train}"
    "++trainer.val_only=${val_only}"
    "++trainer.test_freq=${test_freq}"
    "++trainer.save_freq=${save_freq}"
    "++trainer.total_epochs=${total_epochs}"
    "++trainer.total_training_steps=${total_training_steps}"
    "++trainer.default_local_dir=${CKPTS_DIR}"
    "hydra.run.dir=${OUTPUT_DIR}/hydra_logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
)

# Hydra searchpath: inject absolute path to verl's config directory
HYDRA_ARGS+=("hydra.searchpath=[file://${VERL_EMPO_ROOT}/verl/trainer/config]")

# Optional async mode
ASYNC_MODE=${ASYNC_MODE:-0}
if [ "${ASYNC_MODE}" = "1" ]; then
    HYDRA_ARGS+=("actor_rollout_ref.rollout.mode=async")
    export VLLM_USE_V1=1
    export VERL_VLLM_USE_RAY_BACKEND=1
fi

if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
    HYDRA_ARGS+=("${RESUME_ARGS[@]}")
else
    HYDRA_ARGS+=("trainer.resume_mode=disable")
fi

# ============================================================================
# Dry Run
# ============================================================================
case "${DRY_RUN:-False}" in
    True|true|1|yes|on)
        echo "[Dry Run] Experiment name: ${exp_name}"
        echo "[Dry Run] Output dir: ${OUTPUT_DIR}"
        echo "[Dry Run] Python: ${PYTHON_BIN}"
        printf '[Dry Run] Hydra args:\n'
        printf '  %q\n' "${HYDRA_ARGS[@]}"
        exit 0
        ;;
esac

# ============================================================================
# Run (uses EMPO's original main_dapo entry point)
# ============================================================================
HYDRA_FULL_ERROR=1 "${PYTHON_BIN}" -m recipe.empo.src.main_dapo "${HYDRA_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/console_logs/${exp_name}.log"
PY_EXIT=${PIPESTATUS[0]}

if [ ${PY_EXIT} -ne 0 ]; then
    MISSING_MODULE=$(grep -oP "ModuleNotFoundError: No module named '\K[^']+" "${OUTPUT_DIR}/console_logs/${exp_name}.log" | tail -n 1 || true)
    if [ -n "${MISSING_MODULE}" ]; then
        PKG_NAME="${MISSING_MODULE}"
        if [ "${MISSING_MODULE}" = "pkg_resources" ]; then
            PKG_NAME="setuptools<81"
        fi
        MISSING_REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.missing.txt"
        touch "${MISSING_REQUIREMENTS_FILE}"
        grep -qx "${PKG_NAME}" "${MISSING_REQUIREMENTS_FILE}" || echo "${PKG_NAME}" >> "${MISSING_REQUIREMENTS_FILE}"
        echo "[Info] Added missing module '${PKG_NAME}' to ${MISSING_REQUIREMENTS_FILE}"
    fi
fi

exit ${PY_EXIT}
