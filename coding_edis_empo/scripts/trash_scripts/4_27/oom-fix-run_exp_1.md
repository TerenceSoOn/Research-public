# OOM Fix for `run_exp_1.sh`

**Date:** 2026-04-26
**Script:** `scripts/run_exp_1.sh`
**Backup:** `scripts/run_exp_1.sh.bak`

## Problem

Training crashed at step 1 during the AdamW optimizer step (`torch._foreach_sqrt` in `_multi_tensor_adamw`):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB.
GPU 0 has a total capacity of 47.38 GiB of which 81.75 MiB is free.
45.98 GiB allocated by PyTorch, 738.21 MiB reserved but unallocated.
```

Root cause: vLLM pre-reserved 70% of GPU memory (`gpu_memory_utilization=0.7`, ~33 GiB) for KV cache. Combined with FSDP actor shards and activations, there was zero headroom left when the optimizer needed to materialize temporary tensors.

## Fix Applied

Two changes in `run_exp_1.sh`:

| Parameter | Before | After |
|---|---|---|
| `gpu_memory_utilization` | `0.7` | `0.5` |
| `vllm_enable_sleep_mode` | `False` | `True` |

- Lowering `gpu_memory_utilization` reduces vLLM's KV cache reservation from ~33 GiB to ~24 GiB, freeing ~9 GiB.
- Enabling `sleep_mode` releases vLLM's KV cache memory when the rollout engine is idle (during training), making it available for the optimizer step.

## Result

Step 1 completed successfully with `max_memory_allocated_gb: 38.18` (down from 47.29), providing comfortable headroom.

## Other Options (if OOM recurs)

1. **Reduce `train_micro_batch_size`** (line 223): `4` → `2` or `1`. Directly reduces peak activation memory during training.
2. **Reduce `infer_micro_batch_size`** (line 222): `4` → `2`. Less memory during reference log-prob computation.
3. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** as an environment variable. Helps with fragmentation (the 738 MiB reserved-but-unallocated in the error suggests fragmentation).
4. **Lower `gpu_memory_utilization` further** to `0.4`. Trades rollout throughput for more training headroom.
