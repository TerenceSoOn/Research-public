# Embedding Drift Weighting: Replacing Entropy with Direct Drift Measurement

## Motivation

The current `entropy_high_token_ratio` advantage weighting uses output-level token entropy as a proxy for "distance from original distribution." This proxy degrades after ~50 training steps as entropy collapses to near-zero, making the weighting signal meaningless for 90% of training.

Evidence from the exp3 logs (168 steps, `WEIGHTED=on`):

| Step | actor/entropy | entropy threshold | largest ratio | rest ratio | delta |
|------|--------------|-------------------|---------------|------------|-------|
| 1    | 0.915        | 1.375             | 0.161         | 0.242      | 0.081 |
| 50   | 0.113        | 0.056             | 0.192         | 0.256      | 0.064 |
| 100  | 0.064        | 0.004             | 0.206         | 0.242      | 0.036 |
| 150  | 0.053        | 0.001             | 0.215         | 0.241      | 0.026 |

Despite the signal vanishing, `adv_weight/min=0.5` and `adv_weight/max=1.5` were hit on every single step (168/168), meaning the clips were always saturated by noise.

## Theoretical Basis

Press et al. (2024), "The Entropy Enigma: Success and Failure of Entropy Minimization," shows that entropy minimization has two phases:

- **Phase 1 (beneficial):** EM pulls test embeddings *toward* training embeddings. The model becomes more confident on inputs already close to the training distribution.
- **Phase 2 (harmful):** After many steps, EM pushes embeddings *away* from training embeddings. The model becomes confidently wrong — it collapses to a degenerate distribution far from the original.

The critical insight is in the **embedding space**: cosine similarity between current and original representations increases during Phase 1, then decreases during Phase 2. Output entropy is a downstream consequence of this drift, not a direct measure of it.

## Design Goal

Weight advantages based on how far each response's internal representation has drifted from the pretrained model's distribution center:

- **Correct + close to original center** → amplify (reinforce stable knowledge)
- **Correct + drifted far** → dampen (don't reinforce potentially unstable outputs)
- **Incorrect + drifted far** → amplify (push back toward original center)
- **Incorrect + close to original center** → dampen (don't over-correct stable regions)

This preserves the existing asymmetric weighting design but replaces the degrading entropy proxy with a stable drift signal.

## Proposed Approaches

### Approach A: Reference Model KL as Drift Proxy (Cheapest)

The reference model infrastructure already exists in verl-empo but is currently disabled (`use_kl_loss=False`, `kl_coef=0.0`).

**What it measures:** `KL(pi_current || pi_ref)` per response — how different the current policy's token distribution is from the pretrained model's, aggregated over the response.

**Why it's better than entropy:** KL doesn't collapse. Even when the current policy is very confident (low entropy), KL stays informative because it measures *relative* divergence from the reference, not absolute uncertainty.

**Implementation:**
1. Enable `use_reference_policy` without enabling `use_kl_loss`
2. Compute `ref_log_prob` (already supported in the pipeline)
3. Compute per-response KL from `old_log_prob - ref_log_prob`
4. Use normalized KL as the drift signal in the existing weighting formula

**Cost:** One extra forward pass through the frozen reference model per step (~30-60s for a ~360s step, i.e. 10-15% slowdown).

**Files to modify:**
- `verl-empo/recipe/empo/src/dapo_ray_trainer.py` (weighting function)
- `scripts/run_exp_3.sh`, `scripts/run_exp_4.sh` (config)

### Approach B: Last-Layer Cosine Similarity (Most Aligned with Paper)

Directly implements the paper's Figure 3 insight.

**What it measures:** For each response, extract the last-layer hidden state from both the current actor and the frozen reference model, then compute cosine similarity. Low similarity = high drift.

**Why it's better:** This is the exact metric the paper shows correlates with the Phase 1 → Phase 2 transition.

**Implementation:** Requires modifying actor and reference model forward passes to return hidden states alongside log-probs, then computing per-response mean cosine similarity over response tokens.

**Cost:** Significant memory overhead — storing hidden states (4096-dim for Qwen3-4B) for all response tokens. Estimated ~12.8 GB extra per step for both models. Likely causes OOM on 2-GPU setups.

**Files to modify:**
- Actor worker (`dp_actor.py`)
- Reference worker
- `DataProto` batch structure
- Weighting function in `dapo_ray_trainer.py`

### Approach C: KL with Absolute Threshold (Recommended)

Combines Approach A with an automatic off-switch.

**What it does:**
1. Use per-response KL divergence as the primary drift signal (replaces entropy ratio)
2. Keep the existing weighting asymmetry (correct vs incorrect cluster)
3. Add an absolute threshold: when mean per-response KL drops below a minimum (e.g., 0.01), disable weighting entirely (all weights = 1.0)

**Why this is best:**
- KL is a stable signal that doesn't collapse like entropy
- The absolute threshold provides a clean off-switch, preventing noise-with-clips
- The reference model forward pass is already implemented in the pipeline
- Minimal code changes — reuses existing infrastructure

**Weighting formula:**
```
For each response:
  kl_per_response = mean(old_log_prob - ref_log_prob) over response tokens

If mean(kl_per_response) < min_kl_threshold:
  all weights = 1.0  (weighting is a no-op)
Else:
  For correct (largest-cluster) responses:
    weight = mean_kl / (kl_per_response + eps)
    → low KL (close to pretrained) → high weight
    → high KL (drifted far) → low weight

  For incorrect (rest-cluster) responses:
    weight = kl_per_response / mean_kl
    → high KL (drifted far) → high weight
    → low KL (close to pretrained) → low weight

  Clip weights to [clip_min, clip_max]
```

**New config parameters:**
- `adv_weight_mode: kl_drift_ratio` (new mode)
- `adv_weight_min_kl_threshold: 0.01` (below this, weighting disabled)
- `adv_weight_clip_min: 0.8`
- `adv_weight_clip_max: 1.2`

## Comparison Summary

| Aspect                    | Current (entropy)       | Approach A (KL)         | Approach B (cosine)     | Approach C (KL+threshold) |
|---------------------------|------------------------|------------------------|------------------------|--------------------------|
| Signal stability          | Degrades after ~50 steps | Stable throughout      | Stable throughout      | Stable, auto-disables    |
| Measures drift directly   | No (proxy)             | Partially (output-level)| Yes (embedding-level)  | Partially + safe fallback|
| Extra compute cost        | None                   | +10-15% (ref fwd pass) | +10-15% + memory       | +10-15% (ref fwd pass)   |
| Memory overhead           | None                   | Minimal                | ~12.8 GB               | Minimal                  |
| Code changes              | N/A                    | Small                  | Large                  | Small                    |
| Risk of noise injection   | High (after step 50)   | Low                    | Low                    | Very low (threshold)     |

## Recommendation

**Approach C** (KL with absolute threshold) is recommended. It provides a stable drift signal throughout training, gracefully becomes a no-op when drift is minimal, reuses existing infrastructure, and requires minimal code changes. The 10-15% compute overhead from the reference model forward pass is the main trade-off.

## Implementation Steps

1. Add new `adv_weight_mode: kl_drift_ratio` to `apply_optional_advantage_weighting` in `dapo_ray_trainer.py`
2. Enable reference model forward pass when `adv_weight_mode=kl_drift_ratio` (without activating KL loss)
3. Compute per-response KL from `old_log_prob` and `ref_log_prob` (both already in batch)
4. Implement the weighting formula with absolute KL threshold
5. Add `adv_weight_min_kl_threshold` config parameter
6. Update `run_exp_3.sh` and `run_exp_4.sh` with new config
