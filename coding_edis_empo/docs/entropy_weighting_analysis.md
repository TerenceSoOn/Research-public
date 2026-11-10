# Entropy Weighting Analysis: Why It Degrades and What to Do About It

## Overview

This document analyzes the `entropy_high_token_ratio` advantage weighting mechanism used in exp3/exp4, its theoretical motivation, empirical behavior, and why it fails after early training. It also connects the design to Press et al. (2024) "The Entropy Enigma" and proposes a replacement based on direct drift measurement.

## 1. What the Weighting Does

The weighting is applied in `apply_optional_advantage_weighting()` in `dapo_ray_trainer.py`, after GRPO computes standard advantages.

### Mechanism

1. Compute per-token entropy from the current policy for every response
2. Find a global entropy threshold at the 80th percentile (configurable via `entropy_weight_top_fraction=0.2`) — tokens above this are "high entropy"
3. For each response, compute `high_entropy_ratio = (# high-entropy tokens) / (# total tokens)`
4. Apply asymmetric weighting based on cluster membership:

```
For correct (largest-cluster) responses:
  weight = mean_ratio / (ratio + eps)
  → low ratio (confident) → high weight
  → high ratio (uncertain) → low weight

For incorrect (rest-cluster) responses:
  weight = ratio / mean_ratio
  → high ratio (uncertain) → high weight
  → low ratio (confident) → low weight
```

5. Clip all weights to `[clip_min, clip_max]`
6. Multiply advantages by weights

### Design Intent

The asymmetry targets the original distribution center:
- **Correct + confident** → amplify: reinforce stable knowledge near the pretrained center
- **Correct + uncertain** → dampen: don't over-reinforce noisy correct answers that may be far from center
- **Incorrect + uncertain** → amplify: push harder against uncertain wrong answers
- **Incorrect + confident** → dampen: don't over-correct confidently wrong answers (risks destabilization)

This is motivated by unsupervised RL's core challenge: without ground-truth labels, training should pull the policy toward the densest area of the original distribution while minimizing distribution shift.

## 2. Empirical Behavior (exp3, 168 steps)

Data from `EMPO-qwen3-4b-base-M8_K8_oversampled-off_weighted-on_rw-binary_adv-entropyClusterAware.log`.

### Entropy Collapse

| Step | actor/entropy | entropy threshold | high_entropy_ratio |
|------|--------------|-------------------|--------------------|
| 1    | 0.915        | 1.375             | 0.209              |
| 50   | 0.113        | 0.056             | 0.211              |
| 100  | 0.064        | 0.004             | 0.212              |
| 150  | 0.053        | 0.001             | 0.220              |

By step 100, the entropy threshold is 0.004. The model is nearly deterministic. "High entropy" tokens are those with entropy > 0.001 — the distinction between 99.9% and 99.95% confidence. This is not meaningful uncertainty.

### Weight Distribution

| Metric | All 168 steps |
|--------|---------------|
| `adv_weight/mean` | 1.06 – 1.18 |
| `adv_weight/min` | 0.500 (every step) |
| `adv_weight/max` | 1.500 (every step) |

The clips were saturated on every single step in both directions. This means in every batch, at least one response hit the floor (0.5x) and at least one hit the ceiling (1.5x).

### Per-Cluster Weights

| Step | largest_mean | rest_mean | largest_ratio | rest_ratio | delta |
|------|-------------|-----------|---------------|------------|-------|
| 1    | —           | —         | 0.161         | 0.242      | 0.081 |
| 50   | —           | —         | 0.192         | 0.256      | 0.064 |
| 100  | —           | —         | 0.206         | 0.242      | 0.036 |
| 150  | —           | —         | 0.215         | 0.241      | 0.026 |

The difference in high-entropy ratios between correct and incorrect responses shrinks from 0.081 to 0.026. The weighting formula amplifies this tiny difference into clip-saturating weights.

### Cluster Imbalance

| Step | largest_count | rest_count |
|------|--------------|------------|
| 1    | 210          | 302        |
| 50   | 361          | 151        |
| 100  | 416          | 96         |
| 150  | 404          | 108        |

The majority cluster grows from 41% to 79% of responses. The rest cluster shrinks to ~100 responses, making its weight statistics noisy.

### Score Progression

| Step | critic/score/mean |
|------|-------------------|
| 1    | 0.387             |
| 50   | 0.705             |
| 100  | 0.812             |
| 150  | 0.789             |

Score plateaus around step 100 and slightly regresses — consistent with the weighting injecting noise once the entropy signal is exhausted.

## 3. Why the Entropy Proxy Fails

### Problem 1: Relative threshold never becomes a no-op

The 80th-percentile threshold always splits the token distribution into 80% "low entropy" and 20% "high entropy," regardless of absolute entropy values. At step 150, this means labeling tokens with entropy 0.001 as "high entropy." The weighting formula then computes meaningful-looking weights from meaningless differences.

### Problem 2: The signal vanishes but the weights don't

The high-entropy ratio difference between clusters shrinks to 0.026 by step 150, but the weighting formula (which involves division by ratio + eps) amplifies small differences into large weights that saturate the clips. The result is a bimodal weight distribution: most responses get weight ≈ 1.0, while a few outliers get clipped to the extremes.

### Problem 3: Entropy is a downstream proxy, not a direct measure

This is the deepest issue. The design goal is to measure "distance from the original distribution center." Output entropy measures the model's confidence, which is a *consequence* of distribution drift, not a direct measure of it. By the time entropy has collapsed, the embeddings have already moved. Reweighting based on entropy at that point is adjusting the steering after the car has left the road.

## 4. Connection to Press et al. (2024)

### Paper's Key Finding

Entropy minimization has two phases:
- **Phase 1 (beneficial):** Pulls test embeddings toward training embeddings. Accuracy improves.
- **Phase 2 (harmful):** Pushes embeddings away from training embeddings. Model becomes confidently wrong.

The transition is visible in the embedding space: cosine similarity between current and original representations increases during Phase 1, then decreases during Phase 2 (Figure 3).

### Relevance to Our Setting

In unsupervised RL with EMPO:
- Training naturally reduces entropy (policy sharpens) — analogous to entropy minimization
- We want to stay in Phase 1: converge toward the original distribution center
- We want to avoid Phase 2: overshooting into a degenerate distribution
- The weighting is meant to anchor training to the original center

### The Gap

The paper measures drift directly via embedding cosine similarity. Our weighting uses output entropy as a proxy. The paper shows that the critical signal is in the embedding space, not the output space. Output entropy collapses uniformly (all responses become confident), while embedding drift can vary per response (some responses drift more than others). The entropy proxy loses this per-response discrimination.

## 5. The Clip Range Question

### Previous setting: [0.5, 1.5]

Max weight ratio = 3x (1.5 / 0.5). Saturated every step. Injecting large perturbations from noise.

### Current setting: [0.8, 1.2]

Max weight ratio = 1.5x (1.2 / 0.8). Limits damage but doesn't fix the root cause. The weights will still saturate because the formula amplifies tiny entropy differences.

### Assessment

[0.8, 1.2] is a reasonable harm-reduction measure. It won't hurt and is strictly better than [0.5, 1.5]. But it's a band-aid — the fundamental issue is that the entropy signal is uninformative after ~50 steps, and no clip range can fix that.

## 6. Interaction with GSPO Tight Clipping

Exp1 (no weighting) uses GSPO with `clip_ratio_low=3e-4`, `clip_ratio_high=4e-4` — extremely tight clipping that constrains updates to tiny steps. This acts as the sole regularizer, replacing KL divergence.

Adding advantage weights of [0.5, 1.5] (or even [0.8, 1.2]) on top of this tight clipping creates an uneven update landscape. Some responses get 1.5x the gradient while others get 0.5x, fighting against the carefully balanced trust region. This may explain why exp1 (no weighting) outperforms exp3 (with weighting) — the tight GSPO clip was designed to work with uniform advantages.

## 7. When Weighting Helps vs. Hurts

### Helps (steps 1–50)

- Entropy signal is real and discriminative
- High-entropy ratio difference between clusters is 0.06–0.08
- Weighting provides meaningful anchor toward original center
- Correct-confident responses are genuinely near the pretrained distribution

### Hurts (steps 50+)

- Entropy signal is noise (threshold < 0.01)
- High-entropy ratio difference shrinks to < 0.03
- Weights saturate clips from amplified noise
- Adds variance to gradient without reducing bias

### M=K=8 vs. M=32 K=8

The weighting may be more useful in the oversampled setting (exp4, M=32 K=8) where the larger sample pool maintains more diversity and keeps entropy higher for longer. In the M=K=8 setting (exp3), entropy collapses fast because there's no diversity from oversampling to maintain exploration.

## 8. Recommended Replacement: KL-Based Drift Weighting

See `docs/embedding_drift_weighting_plan.md` for the full proposal. The key idea: replace the entropy proxy with per-response KL divergence from the frozen reference model, which directly measures policy drift and doesn't collapse as entropy does.

## 9. Summary

| Aspect | Finding |
|--------|---------|
| Design intent | Sound — asymmetric weighting toward original center is well-motivated |
| Entropy as proxy | Degrades after ~50 steps due to relative threshold and entropy collapse |
| Clip saturation | 100% of steps hit both clip bounds — weights are bimodal, not smooth |
| Impact on training | Likely neutral-to-harmful after step 50; adds gradient variance without signal |
| Clip range [0.8, 1.2] | Reasonable harm reduction, but doesn't fix root cause |
| Interaction with GSPO | Weighting fights the tight trust region, potentially destabilizing updates |
| Recommendation | Replace entropy proxy with KL-based drift measurement (Approach C) |
