# RL Training Trials (3+ Hours)

Scanned all `.log` files under `/home/guojing/rl/`. Listed below are runs whose first-to-last timestamp spans at least 3 hours.

## verl-entropy-wavelet (April 2026)

| Duration | Log | Time Range |
|---|---|---|
| 72h 35m | `verl-entropy-wavelet/logs/collect_math_20260423_155059.log` | 2026-04-23 15:51 → 2026-04-26 16:26 |
| 72h 12m | `verl-entropy-wavelet/logs/Qwen2.5-Math-1.5B-entropy.log` | 2026-04-23 16:14 → 2026-04-26 16:26 |
| 15h 25m | `verl-entropy-wavelet/logs/collect_data_20260419_231441.log` | 2026-04-19 23:14 → 2026-04-20 14:39 |

> The two 72-hour runs were zombie Ray workers terminated on 2026-04-26.

## EMPO_pjc (Oct 2025 – Jan 2026)

| Duration | Log | Time Range |
|---|---|---|
| 36h 4m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen3_empo_2/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-19 11:26 → 2025-12-20 23:31 |
| 34h 3m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen3_grpo_1/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-21 00:20 → 2025-12-22 10:23 |
| 23h 17m | `EMPO_pjc/EMPO/verl-empo/outputs/console_logs/EMPO-Qwen3-1.7B-redo.log` | 2026-01-18 22:00 → 2026-01-19 21:18 |
| 15h 22m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_empo_2/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-18 19:54 → 2025-12-19 11:17 |
| 12h 38m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen3_empo_1/console_logs/EMPO-Qwen3-1.7B.log` | 2025-12-03 21:18 → 2025-12-04 09:57 |
| 6h 47m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_原脚本_amc23-84/outputs/EMPO-Qwen2.5-Math-1.5B-6.log` | 2025-10-29 08:39 → 2025-10-29 15:26 |
| 6h 8m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_hybrid_2/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-05 10:40 → 2025-12-05 16:49 |
| 5h 38m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_empo_1/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-04 13:01 → 2025-12-04 18:40 |
| 5h 36m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_hybrid_1/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-04 22:36 → 2025-12-05 04:13 |
| 5h 19m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_grpo_1/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-09 10:41 → 2025-12-09 16:01 |
| 3h 12m | `EMPO_pjc/❗IMPORTANT_exp_logs/qwen2.5_grpo_2/console_logs/EMPO-Qwen2.5-1.5B.log` | 2025-12-18 00:28 → 2025-12-18 03:40 |

### Batch size settings

| Log (short name) | train_batch_size | gen_batch_size | ppo_mini_batch_size | ppo_epochs | n | n_samples_per_prompt | n_gpus | total_epochs | reward_type |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_empo_2 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | entropy |
| qwen3_grpo_1 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | gt |
| Qwen3-1.7B-redo | 8 | 8 | 4 | 1 | 16 | 16 | 2 | 1000 | entropy |
| qwen2.5_empo_2 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | entropy |
| qwen3_empo_1 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | entropy |
| qwen2.5_原脚本 | 256 | 256 | 32 | 1 | 16 | 16 | 2 | 1000 | entropy |
| qwen2.5_hybrid_2 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | hybrid:0.1:0.9 |
| qwen2.5_empo_1 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | entropy |
| qwen2.5_hybrid_1 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | hybrid:0.5:0.5 |
| qwen2.5_grpo_1 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | gt |
| qwen2.5_grpo_2 | 8 | 8 | 4 | 1 | 16 | 16 | 4 | 1000 | gt |

## verl / entropy_investigation (Dec 2025 – Jan 2026)

| Duration | Log | Time Range |
|---|---|---|
| 21h 36m | `verl/recipe/entropy_investigation/outputs/Qwen3-1.7B-Base-classic-test2.log` | 2026-01-16 18:21 → 2026-01-17 15:57 |
| 13h 10m | `verl/recipe/entropy_investigation/outputs/Qwen2.5-Math-1.5B-classic-2.log` | 2026-01-14 10:05 → 2026-01-14 23:15 |
| 9h 29m | `verl/recipe/entropy_investigation/outputs/Qwen3-1.7B-Base-classic-aligned2.log` | 2026-01-20 00:33 → 2026-01-20 10:02 |
| 8h 4m | `verl/recipe/entropy_investigation/outputs/Qwen3-1.7B-Base-classic-aligned.log` | 2026-01-17 23:51 → 2026-01-18 07:56 |

### Batch size settings

| Log | train_batch_size | gen_batch_size | ppo_mini_batch_size | ppo_epochs | n (rollout samples) | total_epochs | n_gpus |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base-classic-test2 | 64 | 64 | 8 | 1 | 8 | 2 | 2 |
| Qwen2.5-Math-1.5B-classic-2 | 64 | 64 | 8 | 1 | 8 | 2 | 2 |
| Qwen3-1.7B-Base-classic-aligned2 | 8 | 8 | 4 | 1 | 16 | 2 | 2 |
| Qwen3-1.7B-Base-classic-aligned | 8 | 8 | 4 | 1 | 16 | 2 | 2 |

---

*Generated 2026-04-26. All paths are relative to `/home/guojing/rl/`.*
