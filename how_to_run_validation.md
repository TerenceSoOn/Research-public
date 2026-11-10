# 脚本路径:
`/data/coding/coding_edis_empo/scripts/run_offline_eval_queue.sh`

# checkpoints选择:
脚本第76行, 格式为"tag|checkpoints路径", 多个checkpoints直接加在下一行
```bash
CHECKPOINTS+=("llama-baseline|/data/coding/coding_edis_empo_data/runtime/pretrained/Llama-3.2-3B-Instruct")
CHECKPOINTS+=("tag_2|PATH_TO_CHECKPOINTS_2")
```

# gpu数量设置:
脚本26行:
```bash
GPU_IDS="0,1"
```

# 测试集设置:
脚本27行, 不用测的就删掉
```bash
EVAL_GROUPS="math,nl,coding"
```

# 温度及采样数设置:
脚本31~41行
```bash
MATH_N=16
NL_N=16
LIVECODEBENCH_N=16

# Sampling config. Keep math/NL and LiveCodeBench explicit so eval settings are reproducible.
VAL_DO_SAMPLE=True
VAL_TEMPERATURE=0.2
VAL_TOP_P=0.95
VAL_TOP_K=20
LIVECODEBENCH_TEMPERATURE=0.2
LIVECODEBENCH_TOP_P=0.95
```

# 运行:
```bash
sh /data/coding/coding_edis_empo/scripts/run_offline_eval_queue.sh
```

# outputs:
路径: `/data/coding/coding_edis_empo/eval_runs`

logs目录下可以查看进度