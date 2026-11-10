#!/bin/bash

# 创建必要的目录
mkdir -p /data/coding/EMPO-main/trl/outputs
mkdir -p /data/coding/EMPO-main/trl/outputs/tensorboard_logs
mkdir -p /data/coding/EMPO-main/trl/pretrained

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/data/coding/EMPO-main/trl:$PYTHONPATH"

# vLLM相关设置
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1  # 串行执行CUDA操作以便调试
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 确保设备ID顺序一致
export TOKENIZERS_PARALLELISM=false
# Micro-batch the per-token logprob computation to avoid OOM
export EMPO_LOGITS_MICROBATCH=1

# 禁用 wandb，使用 tensorboard
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# 进入TRL目录
cd /data/coding/EMPO-main/trl

echo "开始EMPO训练，使用Tensorboard监控..."
echo "Tensorboard日志将保存在: /data/coding/EMPO-main/trl/outputs/tensorboard_logs"
echo "模型检查点将保存在: /data/coding/EMPO-main/trl/outputs/Qwen2.5-1.5B-EMPO-NM-COT-20K-Local"

# 启动训练
accelerate launch --config_file recipes/accelerate_configs/no_zero_single_gpu.yaml --num_processes=0 \
    src/open_r1/empo.py \
    --config recipes/Qwen2.5/rl/config_EMPO_NM_COT_20K_1.5B_local.yaml \
    --num_train_epochs=1 \
    --gradient_checkpointing=true
