# OOM Analysis: `run_exp_1.sh` (M8_K8_oversampled-off_weighted-off)

**Date of run:** 2026-04-28 ~00:48 UTC+8
**Date of analysis:** 2026-04-28

## Summary

The run OOM'd on the very first training step (step 0 validation completed successfully).
The crash occurred inside `AdamW.step()` during `update_actor()`.

## Error

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB.
GPU 0 has a total capacity of 23.53 GiB of which 2.50 MiB is free.
Including non-PyTorch memory, this process has 23.48 GiB memory in use.
Of the allocated memory 30.16 GiB is allocated by PyTorch, with 86.50 MiB
allocated in private pools (e.g., CUDA Graphs), and 833.50 MiB is reserved
by PyTorch but unallocated.
```

Stack trace:

```
trainer.fit()
  → actor_rollout_wg.update_actor(batch)
    → actor.update_policy(data)
      → _optimizer_step()
        → AdamW.step()
          → torch._foreach_sqrt(device_exp_avg_sqs)  # OOM here
```

## Hardware

- 3× GPUs (indices 1, 2, 6), 23.53 GiB each (~RTX 3090/4090 24GB)
- FSDP across 3 GPUs (`fsdp_size=-1`)

## Relevant Config

| Parameter | Value | Notes |
|---|---|---|
| `model` | qwen3-4b-base | ~4B params |
| `N_GPUS_PER_NODE` | 3 | |
| `train_prompt_bsz` | 63 | prompts per training batch |
| `n_resp_per_prompt` | 8 | → 504 sequences per batch |
| `train_prompt_mini_bsz` | 8 | → 64 seqs per mini-batch per GPU |
| `train_micro_batch_size` | 3 | |
| `ppo_max_token_len_per_gpu` | 4096 | = max_prompt + max_response |
| `max_prompt_length` | 1024 | |
| `max_response_length` | 3072 | |
| `use_dynamic_bsz` | True | |
| `offload` (param + optimizer) | True | FSDP CPU offload |
| `gpu_memory_utilization` | 0.45 | vLLM rollout KV-cache budget |
| `enable_gradient_checkpointing` | True | |

## Root Cause

The OOM happens during the optimizer step, not during forward/backward. With FSDP
`optimizer_offload=True`, AdamW states (`exp_avg`, `exp_avg_sq`) live on CPU but are
**temporarily moved back to GPU** during `optimizer.step()`. For a 4B-param model
sharded across 3 GPUs, this requires materializing ~5.3GB of optimizer state per GPU
on top of the already-resident model shard and leftover activations/gradients.

The memory budget on a 24GB GPU during the optimizer step looks roughly like:

| Component | Estimate |
|---|---|
| FSDP model shard (fp32 for optimizer) | ~5.3 GB |
| AdamW exp_avg (fp32) | ~5.3 GB |
| AdamW exp_avg_sq (fp32) | ~5.3 GB |
| Gradients (fp32) | ~5.3 GB |
| vLLM KV-cache residual / CUDA context | ~2–3 GB |
| **Total** | **~23–24 GB** |

This leaves essentially zero headroom. The `torch._foreach_sqrt(exp_avg_sqs)` call
needs a temporary buffer of ~130 MiB and there is nothing left.

Contributing factors:

1. **`ppo_max_token_len_per_gpu=4096` is set to exactly one max-length sequence.**
   With `use_dynamic_bsz=True`, shorter sequences (avg response length ~782 tokens)
   get packed together, so micro-batches end up larger than intended, increasing
   activation memory that persists into the optimizer step.

2. **504 sequences per batch across 3 GPUs = 168 sequences per GPU.** This is a
   large per-GPU workload for 24GB cards with a 4B model.

3. **vLLM `gpu_memory_utilization=0.45` during rollout** reserves KV-cache memory
   that may not be fully released before the training phase begins (especially with
   `free_cache_engine=False` and `enable_sleep_mode=True`).

## Suggested Fixes

In order of impact and ease:

1. **Reduce `train_prompt_mini_bsz`** from 8 to 4 (or 2).
   This is the most direct lever — smaller mini-batches mean less activation memory
   carried into the optimizer step.
   ```bash
   train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-4}
   ```

2. **Reduce `train_prompt_bsz`** from 63 to 32.
   Fewer total sequences per training batch.
   ```bash
   train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
   gen_prompt_bsz=${GEN_PROMPT_BSZ:-32}
   ```

3. **Reduce `train_micro_batch_size`** from 3 to 1.
   Fewer sequences per forward/backward micro-step.
   ```bash
   train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE:-1}
   ```

4. **Lower `gpu_memory_utilization`** from 0.45 to 0.35.
   Reduces vLLM's KV-cache reservation, freeing more memory for training.
   ```bash
   gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.35}
   ```

5. **Set `free_cache_engine=True`** to release vLLM KV-cache before training.
   ```
   ++actor_rollout_ref.rollout.free_cache_engine=True
   ```

Start with fix 1 alone; if still OOM, combine with fix 3.
