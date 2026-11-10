#!/bin/bash

# 智能修复 Python 依赖项版本冲突的脚本

# 设置镜像源以加速下载
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=================================================="
echo "Smart Dependency Fixer"
echo "=================================================="
echo "This script will attempt to fix dependency conflicts without re-downloading everything."
echo

# 1. 确保 torch & co. 已安装 (pip 会跳过如果已存在)
echo "--> Step 1: Verifying PyTorch installation..."
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" -i ${PYPI_MIRROR}
echo "PyTorch check complete."
echo

# 2. 解决 vllm 和 sglang 的冲突
# 我们先安装 sglang，它会安装旧版的 xgrammar。
# 然后我们立即安装 vllm，它会把 xgrammar 升级到正确的版本。
echo "--> Step 2: Resolving vllm/sglang conflict..."
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "First, ensuring sglang and its dependencies (including the conflicting xgrammar 0.1.17) are installed..."
pip install "sglang[all]==0.4.6.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python -i ${PYPI_MIRROR}

echo "Now, installing vllm, which will upgrade xgrammar to the required version 0.1.18..."
pip install "vllm==0.8.5.post1" -i ${PYPI_MIRROR}
echo "Conflict resolution step complete."
echo

# 3. 安装 FlashAttention (wget + pip)
echo "--> Step 3: Installing FlashAttention..."
# 检查文件是否已下载
if [ ! -f "flash_attn_correct.whl" ]; then
    echo "Downloading FlashAttention wheel file..."
    wget -nv -O flash_attn_correct.whl https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
else
    echo "FlashAttention wheel file already exists. Skipping download."
fi

pip install flash_attn_correct.whl
# rm flash_attn_correct.whl # You can uncomment this to clean up after installation
echo "FlashAttention installation complete."
echo

echo "=================================================="
echo "Dependency fix script finished."
echo "Please run 'pip list | grep -E \"torch|vllm|sglang|xgrammar\"' to verify the final versions."
echo "=================================================="
