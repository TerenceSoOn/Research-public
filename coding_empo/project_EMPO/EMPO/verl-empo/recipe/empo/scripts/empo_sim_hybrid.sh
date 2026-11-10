#!/usr/bin/env bash
set -x

# If invoked via sh (dash), re-exec with bash so array syntax works
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[Info] Re-executing with bash to support arrays..."
    exec /usr/bin/env bash "$0" "$@"
fi

cd /home/pterence6699/coding/project_EMPO/EMPO/verl-empo/
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# make both local GPUs visible when the variable isn't pre-set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# training topology
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK:-0}"
# export NCCL_NVLS_ENABLE=False
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export VLLM_USE_V1=1

project_name='EMPO'

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=3e-4
clip_ratio_high=4e-4

# Hybrid reward config (EMPO freq + GT correctness)
export HYBRID_ALPHA=${HYBRID_ALPHA:-0.1}
export HYBRID_BETA=${HYBRID_BETA:-0.9}
target="hybrid:${HYBRID_ALPHA}:${HYBRID_BETA}"

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))
enable_overlong_buffer=False
overlong_buffer_len=1024
overlong_penalty_factor=0.0

loss_mode="gspo"
loss_agg_mode="token-mean"

enable_filter_groups=False
confidence_lower_bound=0.3
confidence_upper_bound=1.0
filter_groups_metric=score
max_num_gen_batches=10
train_prompt_bsz=8
gen_prompt_bsz=8
train_prompt_mini_bsz=4
n_resp_per_prompt=16 # note should be divisable by gpu num

exp_name="EMPO-Qwen2.5-1.5B"

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
WORLD_SIZE=$((N_GPUS_PER_NODE * NNODES))
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/home/pterence6699/coding/project_EMPO/ray_data"}
# default path to your local model snapshots (adjust as needed)
MODEL_PATH=${MODEL_PATH:-"/home/pterence6699/coding/project_EMPO/pretrained/Qwen2.5-math-1.5B"}

# Centralized Output Directory
OUTPUT_DIR="/home/pterence6699/coding/project_EMPO/EMPO/verl-empo/outputs"
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_DIR}/checkpoints"}
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard_log"
export WANDB_DIR="/home/pterence6699/coding/project_EMPO/wandb"

# Eval Math Path (for robust GT correctness)
export EVAL_MATH_PATH="/home/pterence6699/coding/project_EMPO/EMPO/eval_math"

# ensure output dirs exist
mkdir -p "${CKPTS_DIR}" "${OUTPUT_DIR}/console_logs" "${TENSORBOARD_DIR}"


RESUME_STEP=${RESUME_STEP:-}
# Default to no resume; set RESUME_PATH or RESUME_STEP explicitly to resume
RESUME_PATH=${RESUME_PATH:-}

if [ -z "${RESUME_PATH}" ] && [ -n "${RESUME_STEP}" ]; then
    RESUME_PATH="${CKPTS_DIR}/global_step_${RESUME_STEP}"
fi

RESUME_ARGS=()
if [ -n "${RESUME_PATH}" ]; then
    if [ ! -d "${RESUME_PATH}" ]; then
        echo "[Error] Resume checkpoint directory ${RESUME_PATH} not found. Set RESUME_PATH to a valid checkpoint before running."
        exit 1
    fi

    ACTOR_DIR="${RESUME_PATH}/actor"
    if [ ! -d "${ACTOR_DIR}" ]; then
        echo "[Error] ${RESUME_PATH} does not contain an actor checkpoint directory. Expected ${ACTOR_DIR}."
        exit 1
    fi

    EXPECTED_SHARD="${ACTOR_DIR}/model_world_size_${WORLD_SIZE}_rank_0.pt"
    if [ ! -f "${EXPECTED_SHARD}" ]; then
        echo "[Error] ${EXPECTED_SHARD} not found. The checkpoint appears to be saved with a different world size." >&2
        echo "        Current world size: ${WORLD_SIZE}. Please resume with the matching number of GPUs first, then re-save." >&2
        exit 1
    fi

    RESUME_ARGS+=("trainer.resume_from_path=${RESUME_PATH}")
    RESUME_ARGS+=("trainer.resume_mode=resume_path")
fi

test_data_dir=/home/pterence6699/coding/project_EMPO/EMPO/math_data

amc23="/home/pterence6699/coding/project_EMPO/EMPO/math_data/amc23_1010.parquet"

train_path=/home/pterence6699/coding/project_EMPO/EMPO/math_data/NM_20K_1010.parquet

TRAIN_FILES="['$train_path']"
TEST_FILES="['$amc23']"
# TEST_FILES="['$amc23']"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=True

HYDRA_ARGS=(
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${TEST_FILES}"
    "data.use_shm=False"
    "data.prompt_key=prompt"
    "data.truncation='left'"
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    "data.gen_batch_size=${gen_prompt_bsz}"
    "data.train_batch_size=${train_prompt_bsz}"
    "data.filter_overlong_prompts=False"
    "actor_rollout_ref.rollout.n=${n_resp_per_prompt}"
    "actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}"
    "actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}"
    "actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}"
    "actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}"
    "actor_rollout_ref.actor.clip_ratio_c=10.0"
    "actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}"
    "algorithm.adv_estimator=${adv_estimator}"
    "algorithm.use_kl_in_reward=${use_kl_in_reward}"
    "algorithm.kl_ctrl.kl_coef=${kl_coef}"
    "algorithm.filter_groups.enable=${enable_filter_groups}"
    "algorithm.filter_groups.metric=${filter_groups_metric}"
    "algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}"
    "algorithm.filter_groups.confidence_lower_bound=${confidence_lower_bound}"
    "algorithm.filter_groups.confidence_upper_bound=${confidence_upper_bound}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.use_fused_kernels=False"
    "actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "+actor_rollout_ref.model.override_config.attention_dropout=0."
    "+actor_rollout_ref.model.override_config.embd_pdrop=0."
    "+actor_rollout_ref.model.override_config.resid_pdrop=0."
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.use_liger=False"
    "actor_rollout_ref.actor.optim.lr=5e-7"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
    "actor_rollout_ref.actor.optim.weight_decay=0"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}"
    "actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${offload}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.expert_parallel=False"
    "actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.rollout.temperature=${temperature}"
    "actor_rollout_ref.rollout.top_p=${top_p}"
    # Async/server mode can be enabled via ASYNC_MODE=1 below
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.rollout.val_kwargs.n=16"
    "actor_rollout_ref.rollout.val_kwargs.top_k=20"
    "actor_rollout_ref.rollout.val_kwargs.top_p=0.95"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.6"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${offload}"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
    "reward_model.reward_manager=empo"
    "reward_model.reward_kwargs.reward_type=${target}"
    "reward_model.overlong_buffer.enable=${enable_overlong_buffer}"
    "reward_model.overlong_buffer.len=${overlong_buffer_len}"
    "reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}"
    "reward_model.use_xverify=False"
    "reward_model.use_eval_math=True"
    "trainer.logger=['console','tensorboard']"
    "trainer.project_name=${project_name}"
    "trainer.experiment_name=${exp_name}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.val_before_train=True"
    "trainer.val_only=False"
    "trainer.test_freq=10"
    "trainer.save_freq=100"
    "trainer.total_epochs=1000"
    "trainer.total_training_steps=500"
    "trainer.default_local_dir=${CKPTS_DIR}"
    "hydra.run.dir=${OUTPUT_DIR}/hydra_logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
)

# Optional toggle for async/server mode rollout (vLLM AsyncLLM)
ASYNC_MODE=${ASYNC_MODE:-0}
if [ "${ASYNC_MODE}" = "1" ]; then
    HYDRA_ARGS+=("actor_rollout_ref.rollout.mode=async")
    # Ensure new vLLM API and ray backend are used
    export VLLM_USE_V1=1
    export VERL_VLLM_USE_RAY_BACKEND=1
fi

if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
    HYDRA_ARGS+=("${RESUME_ARGS[@]}")
else
    # No explicit resume provided: force fresh training and avoid auto-resume
    HYDRA_ARGS+=("trainer.resume_mode=disable")
fi

HYDRA_FULL_ERROR=1 python3 -m recipe.empo.src.main_dapo "${HYDRA_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/console_logs/${exp_name}.log"