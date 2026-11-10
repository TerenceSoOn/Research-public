# OOM Analysis for `run_exp_1.sh`

**Date:** 2026-04-27
**Script:** `scripts/run_exp_1.sh`
**Experiment:** `M8_K8_oversampled-off_weighted-off_rw-binary` (Llama-3.2-3B-Instruct)
**Log:** `outputs/M8_K8_oversampled-off_weighted-off_rw-binary/console_logs/EMPO-Llama-3.2-3B-Instruct-M8_K8_oversampled-off_weighted-off_rw-binary.log`

## GPU Configuration

- **GPU count:** 1 (`N_GPUS_PER_NODE=1`)
- **CUDA_VISIBLE_DEVICES:** `6`
- **GPU capacity:** 47.38 GiB (likely A40 or L40)
- **`gpu_memory_utilization`:** 0.5 (vLLM KV cache reserves ~24 GiB)
- **`enable_sleep_mode`:** True
- **`param_offload` / `optimizer_offload`:** True (intended)

## Crash Details

- **When:** Step 1 (first training step), immediately after validation at step 0 completed successfully.
- **Where:** `update_actor` → `optimizer.step()` → AdamW `_init_group` → `torch.zeros_like()` for `exp_avg_sq`
- **Allocation that failed:** 386 MiB

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 386.00 MiB.
GPU 0 has a total capacity of 47.38 GiB of which 331.75 MiB is free.
Including non-PyTorch memory, this process has 47.00 GiB memory in use.
Of the allocated memory 55.14 GiB is allocated by PyTorch, with 785.58 MiB
allocated in private pools (e.g., CUDA Graphs), and 1.12 GiB is reserved
by PyTorch but unallocated.
```

## Stack Trace (abbreviated)

```
dapo_ray_trainer.py:700  trainer.fit() → actor_output = self.actor_rollout_wg.update_actor(batch)
fsdp_workers.py:613      update_actor → self.actor.update_policy(data=data)
dp_actor.py:457           update_policy → grad_norm = self._optimizer_step()
dp_actor.py:259           _optimizer_step → self.actor_optimizer.step()
torch/optim/adamw.py:175  _init_group → state["exp_avg_sq"] = torch.zeros_like(...)
→ torch.OutOfMemoryError
```

## Root Cause

AdamW lazily initializes its state buffers (`exp_avg`, `exp_avg_sq`) on the first `optimizer.step()`. For a 3B-parameter model this requires ~12 GiB (two FP32 copies of all parameters). At that moment the GPU was already nearly full:

| Consumer | Estimated |
|---|---|
| vLLM KV cache (50% reservation) | ~24 GiB |
| FSDP actor shards + activations | ~20 GiB |
| CUDA Graphs private pools | ~0.8 GiB |
| Reserved but unallocated (fragmentation) | ~1.1 GiB |
| **Free at crash time** | **0.33 GiB** |

Even though `sleep_mode=True` should release the KV cache during training, the optimizer init happened before (or concurrently with) the cache release, leaving no headroom.

### Suspicious Config Override

The Hydra-resolved config printed in the log shows **two** actor config blocks:

1. **Your overrides (applied):** `param_offload: True`, `optimizer_offload: True`, `ppo_max_token_len_per_gpu: 4096`
2. **Base/default config (lines ~165-191):** `param_offload: False`, `optimizer_offload: False`, `ppo_max_token_len_per_gpu: 32768`, model path `~/models/deepseek-llm-7b-chat`

The second block appears to be a stale default that may be partially overriding or conflicting with the intended offload settings. If `optimizer_offload` is not actually active, the AdamW states remain on GPU and OOM is inevitable on a single 48 GiB card.

### Contributing Factor: Fragmentation

The error reports 1.12 GiB reserved-but-unallocated by PyTorch, indicating memory fragmentation. The error message itself suggests setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Recommendations

1. **Verify offload is actually active** — add a debug print in `dp_actor.py` to confirm `optimizer_offload=True` is reaching the FSDP worker. The stale default config block is suspicious.
2. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — reclaims the ~1.1 GiB lost to fragmentation.
3. **Lower `gpu_memory_utilization` further** to `0.4` or `0.35` — trades rollout throughput for ~5-7 GiB more training headroom.
4. **Reduce `train_prompt_bsz`** from 64 to 32 — fewer sequences in the actor update means lower peak activation memory.
5. **Reduce `ppo_mini_batch_size`** from 8 to 4 — smaller micro-batches during the PPO update.
