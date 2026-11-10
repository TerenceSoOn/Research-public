#!/bin/bash

# Tensorboard 监控脚本
echo "启动 Tensorboard 监控..."
echo "日志目录: /data/coding/EMPO-main/trl/outputs/tensorboard_logs"
echo "访问地址: http://localhost:6006"

# 启动 Tensorboard
tensorboard --logdir=/data/coding/EMPO-main/trl/outputs/tensorboard_logs --port=6006 --host=0.0.0.0
