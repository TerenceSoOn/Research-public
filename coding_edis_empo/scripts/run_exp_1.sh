#!/usr/bin/env bash
set -x

# If invoked via sh (dash), re-exec with bash so array syntax works.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[Info] Re-executing with bash to support arrays..."
    exec /usr/bin/env bash "$0" "$@"
fi

# ============================================================================
# Directory Setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to verl-empo (EMPO repo) - adjust if your layout differs.
VERL_EMPO_ROOT="${VERL_EMPO_ROOT:-$(cd "${PROJECT_ROOT}/../coding_empo/project_EMPO/EMPO/verl-empo" && pwd)}"
EMPO_ROOT="${EMPO_ROOT:-$(cd "${VERL_EMPO_ROOT}/.." && pwd)}"
PROJECT_EMPO_ROOT="${PROJECT_EMPO_ROOT:-$(cd "${VERL_EMPO_ROOT}/../.." && pwd)}"

# Model selection: edit MODEL_PATH directly when switching models.
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/runtime/pretrained/qwen3-4b-base}"
MODEL_TAG="${MODEL_TAG:-qwen3-4b-base}"

source "${SCRIPT_DIR}/lib/runtime_env.sh"
resolve_train_venv

cd "${PROJECT_ROOT}"

# PYTHONPATH: this repo first, then verl-empo.
export PYTHONPATH="${PROJECT_ROOT}:${VERL_EMPO_ROOT}:${PYTHONPATH:-}"

source "${VENV_PATH}/bin/activate"

# ============================================================================
# Environment Variables
# ============================================================================
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
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
# EMPO Binary 2x2 Experiment Switches
#
# Edit only this block for the four primary experiments:
#   OVERSAMPLED=off, WEIGHTED=off -> M=8,  K=8, binary reward
#   OVERSAMPLED=on,  WEIGHTED=off -> M=32, random K=8, binary reward
#   OVERSAMPLED=off, WEIGHTED=on  -> M=8,  K=8, binary reward + entropy adv weighting
#   OVERSAMPLED=on,  WEIGHTED=on  -> M=32, random K=8, binary reward + entropy adv weighting
# ============================================================================
OVERSAMPLED=off
WEIGHTED=off

SELECTED_TOP_K=8
OVERSAMPLE_RATIO=4
REWARD_TYPE=voting
ENTROPY_WEIGHT_TOP_FRACTION=0.2
ADV_WEIGHT_CLIP_MIN=0.5
ADV_WEIGHT_CLIP_MAX=1.5

case "${OVERSAMPLED}" in
    on)
        selected_top_k="${SELECTED_TOP_K}"
        oversample_ratio="${OVERSAMPLE_RATIO}"
        if ! [[ "${oversample_ratio}" =~ ^[0-9]+$ ]] || [ "${oversample_ratio}" -lt 1 ]; then
            echo "[Error] OVERSAMPLE_RATIO must be a positive integer, got ${oversample_ratio}." >&2
            exit 1
        fi
        n_resp_per_prompt=$((selected_top_k * oversample_ratio))
        ;;
    off)
        selected_top_k="${SELECTED_TOP_K}"
        oversample_ratio=1
        n_resp_per_prompt="${selected_top_k}"
        ;;
    *)
        echo "[Error] OVERSAMPLED must be on or off, got ${OVERSAMPLED}." >&2
        exit 1
        ;;
esac

case "${WEIGHTED}" in
    on)
        adv_weight_mode="entropy_high_token_ratio"
        adv_tag="_adv-entropyClusterAware"
        ;;
    off)
        adv_weight_mode="off"
        adv_tag=""
        ;;
    *)
        echo "[Error] WEIGHTED must be on or off, got ${WEIGHTED}." >&2
        exit 1
        ;;
esac

if ! [[ "${selected_top_k}" =~ ^[0-9]+$ ]] || [ "${selected_top_k}" -lt 1 ]; then
    echo "[Error] SELECTED_TOP_K must be a positive integer, got ${selected_top_k}." >&2
    exit 1
fi
if [ "${selected_top_k}" -gt "${n_resp_per_prompt}" ]; then
    echo "[Error] SELECTED_TOP_K=${selected_top_k} cannot exceed M=${n_resp_per_prompt}." >&2
    exit 1
fi

target="${REWARD_TYPE}"
selection_family="random_topk"
balanced_selection_method="D"
reward_method="use_binary_reward"
group_largest_count=""
group_rest_count=""
underfilled_policy="skip_prompt"
entropy_weight_top_fraction="${ENTROPY_WEIGHT_TOP_FRACTION}"
adv_weight_clip_min="${ADV_WEIGHT_CLIP_MIN}"
adv_weight_clip_max="${ADV_WEIGHT_CLIP_MAX}"
use_wandb="${USE_WANDB:-True}"
wandb_mode="${WANDB_MODE:-online}"
wandb_group="EMPO-Binary-2x2"

case "${target}" in
    voting)
        ;;
    *)
        echo "[Error] This 2x2 script intentionally supports only REWARD_TYPE=voting." >&2
        exit 1
        ;;
esac

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

# --- Training batch sizes ---
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-8}

# --- Sequence lengths ---
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 1))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 3))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-1024}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-0.0}
min_response_length=${MIN_RESPONSE_LENGTH:-128}

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
topk_checkpoint_enable=${TOPK_CHECKPOINT_ENABLE:-True}
topk_checkpoint_k=${TOPK_CHECKPOINT_K:-2}
topk_checkpoint_metric=${TOPK_CHECKPOINT_METRIC:-val-core/amc/acc/mean@${val_n}}
topk_checkpoint_dir_name=${TOPK_CHECKPOINT_DIR_NAME:-top_amc_checkpoints}

# --- Training schedule ---
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-200}
total_epochs=${TOTAL_EPOCHS:-1000}
total_training_steps=${TOTAL_TRAINING_STEPS:-2500}
val_before_train=${VAL_BEFORE_TRAIN:-True}
val_only=${VAL_ONLY:-False}

# --- Dynamic batching ---
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
infer_micro_batch_size=${INFER_MICRO_BATCH_SIZE:-2}
train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE:-2}
offload=${OFFLOAD:-True}

# --- GPU memory ---
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.7}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}
vllm_enable_sleep_mode=${VLLM_ENABLE_SLEEP_MODE:-True}

# ============================================================================
# Experiment Naming
# ============================================================================
project_name="EMPO-Binary-2x2"
run_signature="M${n_resp_per_prompt}_K${selected_top_k}_oversampled-${OVERSAMPLED}_weighted-${WEIGHTED}_rw-binary${adv_tag}"
exp_name="EMPO-${MODEL_TAG}-${run_signature}"

# ============================================================================
# Ray
# ============================================================================
RAY_ADDRESS=${RAY_ADDRESS:-local}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
WORLD_SIZE=$((N_GPUS_PER_NODE * NNODES))
export RAY_DISABLE_DASHBOARD=${RAY_DISABLE_DASHBOARD:-1}

# ============================================================================
# Model & Data Paths
# ============================================================================
resolve_runtime_defaults

OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs"}
OUTPUT_DIR="${OUTPUT_ROOT}/${run_signature}"
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_DIR}/checkpoints"}
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard_log"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_MODE="${wandb_mode}"
export WANDB_GROUP="${wandb_group}"

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
# Data Paths (online training validates on AMC23 only)
# ============================================================================
test_data_dir="${DATA_ROOT}"

# Training data
train_path="${TRAIN_DATA:-${DATA_ROOT}/NM_20K_1010.parquet}"

# Validation data
amc23="${test_data_dir}/amc23_1010.parquet"
if [ -n "${VAL_FILES:-}" ] && [ "${VAL_FILES}" != "amc23" ]; then
    echo "[Error] Online training scripts validate on AMC23 only. Use offline eval for VAL_FILES=${VAL_FILES}." >&2
    exit 1
fi
TEST_FILES="['$amc23']"

TRAIN_FILES="['$train_path']"

echo "===== EMPO Binary 2x2 Configuration ====="
echo "  OVERSAMPLED:              ${OVERSAMPLED}"
echo "  WEIGHTED:                 ${WEIGHTED}"
echo "  M (generated):            ${n_resp_per_prompt}"
echo "  K (selected):             ${selected_top_k}"
echo "  oversample_ratio:         ${oversample_ratio}"
echo "  selection_family:         ${selection_family} (fixed random selection)"
echo "  reward_method:            ${reward_method} (fixed binary 1/0 tag)"
echo "  adv_weight_mode:          ${adv_weight_mode}"
echo "  entropy_weight_top_fraction: ${entropy_weight_top_fraction}"
echo "  adv_weight_clip_min:      ${adv_weight_clip_min}"
echo "  adv_weight_clip_max:      ${adv_weight_clip_max}"
echo "  first_round_vote_mode:    majority (fixed)"
echo "  trainer_logger:           ${trainer_logger}"
echo "  use_wandb:                ${use_wandb}"
echo "  wandb_mode:               ${wandb_mode}"
echo "  wandb_project:            ${project_name}"
echo "  wandb_group:              ${wandb_group}"
echo "  reward_type:              ${target}"
echo "  temperature:              ${temperature}"
echo "  learning_rate:            ${learning_rate}"
echo "  rollout_enforce_eager:    ${rollout_enforce_eager}"
echo "  vllm_enable_sleep_mode:   ${vllm_enable_sleep_mode}"
echo "  cuda_visible:             ${CUDA_VISIBLE_DEVICES}"
echo "  ray_address:              ${RAY_ADDRESS}"
echo "  model:                    ${MODEL_PATH}"
echo "  train_data:               ${train_path}"
echo "  val_data:                 ${TEST_FILES}"
echo "  output_dir:               ${OUTPUT_DIR}"
echo "  topk_ckpt_enable:         ${topk_checkpoint_enable}"
echo "  topk_ckpt_k:              ${topk_checkpoint_k}"
echo "  topk_ckpt_metric:         ${topk_checkpoint_metric}"
echo "  topk_ckpt_dir:            ${topk_checkpoint_dir_name}"
echo "================================================"

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
    "++actor_rollout_ref.rollout.calculate_log_probs=True"
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
    "++actor_rollout_ref.model.use_fused_kernels=True"
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
    "++actor_rollout_ref.rollout.max_num_batched_tokens=$((2 * (max_prompt_length + max_response_length)))"
    "++actor_rollout_ref.rollout.temperature=${temperature}"
    "++actor_rollout_ref.rollout.top_p=${top_p}"
    "++actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}"
    "++actor_rollout_ref.rollout.val_kwargs.n=${val_n}"
    "++actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}"
    "++actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
    "++actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}"
    "++actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}"
    "++actor_rollout_ref.rollout.enable_sleep_mode=${vllm_enable_sleep_mode}"
    "++actor_rollout_ref.rollout.free_cache_engine=False"
    "++actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size}"
    "++actor_rollout_ref.ref.fsdp_config.param_offload=${offload}"
    "++actor_rollout_ref.ref.ulysses_sequence_parallel_size=1"
    "++actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
    # Binary EMPO 2x2 reward config.
    "++reward_model.reward_manager=oversampled_empo"
    "++reward_model.reward_kwargs.reward_type=${target}"
    "++reward_model.reward_kwargs.n_samples_per_prompt=${n_resp_per_prompt}"
    "++reward_model.reward_kwargs.n_votes_per_prompt=${n_resp_per_prompt}"
    "++reward_model.reward_kwargs.selected_top_k=${selected_top_k}"
    "++reward_model.reward_kwargs.oversample_ratio=${oversample_ratio}"
    "++reward_model.reward_kwargs.selection_family=${selection_family}"
    "++reward_model.reward_kwargs.balanced_selection_method=${balanced_selection_method}"
    "++reward_model.reward_kwargs.reward_method=${reward_method}"
    "++reward_model.reward_kwargs.largest_group_count=${group_largest_count:-null}"
    "++reward_model.reward_kwargs.rest_group_count=${group_rest_count:-null}"
    "++reward_model.reward_kwargs.underfilled_policy=${underfilled_policy}"
    "++reward_model.reward_kwargs.adv_weight_mode=${adv_weight_mode}"
    "++reward_model.reward_kwargs.entropy_weight_top_fraction=${entropy_weight_top_fraction}"
    "++reward_model.reward_kwargs.adv_weight_clip_min=${adv_weight_clip_min}"
    "++reward_model.reward_kwargs.adv_weight_clip_max=${adv_weight_clip_max}"
    "++reward_model.reward_kwargs.min_response_length=${min_response_length}"
    "++reward_model.overlong_buffer.enable=${enable_overlong_buffer}"
    "++reward_model.overlong_buffer.len=${overlong_buffer_len}"
    "++reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}"
    "++reward_model.use_xverify=False"
    "++reward_model.use_eval_math=True"
    "++trainer.logger=${trainer_logger}"
    "++trainer.project_name=${project_name}"
    "++trainer.experiment_name=${exp_name}"
    "++trainer.wandb_group=${wandb_group}"
    "++trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "++trainer.nnodes=${NNODES}"
    "++trainer.val_before_train=${val_before_train}"
    "++trainer.val_only=${val_only}"
    "++trainer.test_freq=${test_freq}"
    "++trainer.save_freq=${save_freq}"
    "++trainer.topk_checkpoint_enable=${topk_checkpoint_enable}"
    "++trainer.topk_checkpoint_k=${topk_checkpoint_k}"
    "++trainer.topk_checkpoint_metric=${topk_checkpoint_metric}"
    "++trainer.topk_checkpoint_dir_name=${topk_checkpoint_dir_name}"
    "++trainer.total_epochs=${total_epochs}"
    "++trainer.total_training_steps=${total_training_steps}"
    "++trainer.default_local_dir=${CKPTS_DIR}"
    "hydra.run.dir=${OUTPUT_DIR}/hydra_logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
)

# Hydra searchpath: inject absolute path to verl's config directory.
HYDRA_ARGS+=("hydra.searchpath=[file://${VERL_EMPO_ROOT}/verl/trainer/config]")

# Optional async mode.
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
# Run
# ============================================================================
SCRIPT_LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_LOG_DIR}"
HYDRA_FULL_ERROR=1 "${PYTHON_BIN}" -m src.main "${HYDRA_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/console_logs/${exp_name}.log" "${SCRIPT_LOG_DIR}/${exp_name}.log"
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
