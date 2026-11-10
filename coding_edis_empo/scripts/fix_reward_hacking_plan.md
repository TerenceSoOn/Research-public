# Fix Reward Hacking in Exp 3 (Entropy-Weighted Advantage)

## Diagnosis Summary

Exp 3 (`OVERSAMPLED=off, WEIGHTED=on`) completed 200 steps in ~6h but the model collapsed:

| Metric | Step 1 | Step 20 | Step 50 | Step 100 | Step 200 |
|---|---|---|---|---|---|
| actor/entropy | 3.57 | 0.53 | 0.15 | 0.10 | 0.09 |
| response_length/mean | 592 | 339 | 169 | 115 | 65 |
| val-core/amc/acc/mean@16 | 0.242 | -- | -- | 0.039 | 0.053 |
| critic/score/mean (train reward) | 0.35 | 0.50 | 0.47 | 0.60 | 0.74 |

For comparison, exp 1 (no weighting) at step 100: entropy=0.25, response_length=639, val-acc=0.283.

The core failure mode: the model learned to produce very short, low-entropy responses that game the binary reward. Training reward goes up while validation accuracy collapses — textbook reward hacking.

## How the Weighting Strategy Works

Source: `verl-empo/recipe/empo/src/dapo_ray_trainer.py`, function `apply_optional_advantage_weighting` (lines 46-168).

This function runs after GRPO computes per-sample advantages. It rescales those advantages based on each response's token-level entropy profile. The intent is to upweight "confident correct" responses and downweight "confident incorrect" ones. Here is the step-by-step logic:

### Step 1: Compute the high-entropy threshold

```python
quantile = 1.0 - entropy_weight_top_fraction   # default: 1.0 - 0.2 = 0.8
threshold = torch.quantile(valid_entropies, quantile)
```

Collect all token-level entropies across the batch (using `entropy_reference_mask` which covers all M generated responses, not just the K selected ones). Take the 80th percentile as the "high entropy" threshold. Any token with entropy >= this threshold is considered a "high-entropy token."

### Step 2: Compute per-response high-entropy ratio

```python
ratio_i = (number of high-entropy tokens in response i) / (total valid tokens in response i)
```

This gives each response a scalar `ratio` in [0, 1] representing what fraction of its tokens are "uncertain." A low ratio means the model was very confident throughout the response. A high ratio means many tokens had above-average entropy.

### Step 3: Compute weights (different for largest-cluster vs rest)

```python
mean_ratio = mean(ratios across all non-empty responses)

# For largest-cluster (majority / binary-positive) samples:
stable_weight = mean_ratio / (ratio + 1e-8)

# For rest-cluster (minority / binary-zero) samples:
unstable_weight = ratio / mean_ratio
```

There are two dimensions (cluster x entropy level), giving 4 cases:

| | Low entropy ratio (< mean) | High entropy ratio (> mean) |
|---|---|---|
| **Largest cluster** (correct) | `stable_weight > 1` — upweighted | `stable_weight < 1` — downweighted |
| **Rest cluster** (incorrect) | `unstable_weight < 1` — downweighted | `unstable_weight > 1` — upweighted |

The intuition for each case:
- **Largest + low entropy** (confident correct): upweighted — reinforce clean positive signal
- **Largest + high entropy** (uncertain correct): downweighted — noisy positive signal, learn less from it
- **Rest + low entropy** (confident wrong): downweighted — the model is confidently wrong, don't amplify that gradient
- **Rest + high entropy** (uncertain wrong): upweighted — the model is unsure and wrong, this is informative for learning

### Step 4: Clip and apply

```python
stable_weights = clamp(stable_weights, clip_min=0.5, clip_max=1.5)
unstable_weights = clamp(unstable_weights, clip_min=0.5, clip_max=1.5)

advantages = advantages * effective_weights  # element-wise per sample
```

The clipping bounds the weight to [0.5, 1.5], so the most-weighted sample gets at most 3x the advantage of the least-weighted sample.

### What the logged metrics mean

| Metric | Meaning |
|---|---|
| `adv_weight/mean` | Average effective weight across selected training samples |
| `adv_weight/high_entropy_ratio_mean` | Average fraction of high-entropy tokens per response |
| `adv_weight/high_entropy_threshold` | The 80th-percentile entropy value used as cutoff |
| `adv_weight/largest_mean` | Average weight for majority-cluster (correct) samples |
| `adv_weight/rest_mean` | Average weight for rest-cluster (incorrect) samples |
| `adv_weight/largest_count` | Number of samples in the majority cluster |
| `adv_weight/rest_count` | Number of samples in the rest cluster |

### How the weighting evolved during exp3

| Step | hi_ent_threshold | hi_ent_ratio | largest_w | rest_w | largest_n | rest_n |
|---|---|---|---|---|---|---|
| 1 | 10.188 | 0.054 | 1.423 | 0.617 | 181 | 331 |
| 10 | 2.828 | 0.054 | 1.406 | 0.765 | 232 | 280 |
| 20 | 0.648 | 0.164 | 1.182 | 0.960 | 255 | 257 |
| 50 | 0.203 | 0.195 | 1.148 | 0.997 | 238 | 274 |
| 100 | 0.073 | 0.203 | 1.109 | 1.050 | 307 | 205 |
| 150 | 0.056 | 0.181 | 1.137 | 1.082 | 339 | 173 |
| 200 | 0.030 | 0.160 | 1.116 | 1.149 | 378 | 134 |

Key observations:
- The `high_entropy_threshold` collapsed from 10.2 to 0.03 — the batch entropy distribution compressed dramatically.
- `largest_count` grew from 181 to 378 (out of 512) — the majority cluster absorbed almost everything as the model converged to a single short-answer mode.
- The weight differential (`largest_w` vs `rest_w`) actually narrowed over time, but by then the damage was done — the entropy had already collapsed by step 20-30.

## Root Cause Analysis

The advantage weighting mechanism creates a positive feedback loop:

1. Early on, the model produces diverse responses with high entropy. The weighting correctly upweights confident correct answers.
2. As training progresses, the model starts favoring short, confident responses (they get higher weights). Entropy drops.
3. As entropy drops, the threshold drops too (it's a batch-relative percentile). The "high entropy" bar keeps lowering.
4. With near-zero entropy everywhere, almost all responses land in the majority cluster with similar low-entropy profiles. The weighting loses its discriminative power, but the policy has already collapsed.

The critical window is steps 5-30, where entropy drops from ~3.5 to ~0.3. After that, recovery is essentially impossible without external regularization.

Three compounding factors:

1. **No entropy regularization**: `entropy_coeff=0` in `run_exp_3.sh` line 400. Nothing prevents entropy from collapsing to near-zero.
2. **No KL penalty**: `use_kl_loss=False`, `kl_coef=0.0` (lines 178-181). The model can drift arbitrarily far from the pretrained policy.
3. **Loose advantage weight clipping**: `[0.5, 1.5]` allows a 3x ratio between the most and least weighted samples, enough to drive the collapse.

### Feedback Loop Diagram

```
Model produces short low-entropy responses
        |
        v
Binary reward stays high for short correct answers
        |
        v
Advantage weighting upweights low-entropy majority-cluster samples
        |
        v
Policy update reinforces short confident outputs
        |
        v
Entropy drops further  --->  Response length shrinks  --->  Validation accuracy collapses
        |
        +--- loops back to top ---+
```

## Proposed Fixes (Ranked by Priority)

### ~~Fix 1: Add entropy bonus to the policy loss~~ [SKIPPED]

~~In `run_exp_3.sh` line 400, change `entropy_coeff=0` to `entropy_coeff=0.001`.~~

Not applying — the weighting strategy itself works well on other models (e.g. Qwen2.5-1.5B-math). The collapse is better addressed by tightening the clipping range.

### Fix 2: Tighten advantage weight clipping [APPLYING]

In `run_exp_3.sh` lines 75-76, change:

```bash
ADV_WEIGHT_CLIP_MIN=0.5
ADV_WEIGHT_CLIP_MAX=1.5
```

to:

```bash
ADV_WEIGHT_CLIP_MIN=0.8
ADV_WEIGHT_CLIP_MAX=1.2
```

This reduces the max weight ratio from 3x to 1.5x, limiting how aggressively the weighting can push the policy toward low-entropy outputs.

### ~~Fix 3: Add mild KL regularization~~ [SKIPPED]

~~In `run_exp_3.sh` lines 180-181, enable `use_kl_loss=True` with `kl_loss_coef=0.01`.~~

Not applying.

### ~~Fix 4 (optional): Add early-stopping on entropy or val-acc~~ [SKIPPED]

~~Add a monitoring check in the training loop.~~

Not applying.

## Recommended Re-run Strategy

Apply Fix 2 (tighter clipping: `[0.8, 1.2]`) and Fix 5 (filter short responses < 128 tokens from selection).

### Fix 5: Filter short responses from selection [APPLYING]

Responses shorter than 128 tokens are excluded from the candidate pool in `_select_local_indices` before selection. They still get generated and scored (for metrics), but can never be selected for training.

Changes made:
- `src/edis_reward_manager.py`: Added `min_response_length` parameter to `EmpoBinaryRewardManager.__init__`, built an `eligible_mask` from `response_lengths` before calling `_select_local_indices`, and updated `_select_local_indices` to filter by the mask. Added `num_short_filtered` counter to console log and `empo_info`.
- `src/config/edis_empo_trainer.yaml`: Added `min_response_length: 0` default.
- `scripts/run_exp_3.sh`: Added `min_response_length=128` variable and `++reward_model.reward_kwargs.min_response_length` hydra arg.
- `verl-empo/recipe/empo/src/dapo_ray_trainer.py`: Extracts `num_short_filtered` from `empo_info` and logs it as `selection/num_short_filtered` metric.
