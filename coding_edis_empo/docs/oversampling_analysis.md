# Oversampling as Density Estimation: Why M > K Works

## Core Thesis

Oversampling (M=32, K=8) is a variance reduction technique for mode estimation. Its value comes from better identifying the densest region of the model's output distribution, not from selecting "better" training samples. Training toward the densest region improves performance regardless of whether that region is correct, because it organizes the model's internal representations — the Phase 1 mechanism described in Press et al. (2024).

## Background

In unsupervised RL with EMPO, there are no ground-truth labels. The reward comes from majority voting over the model's own responses. The training objective is to sharpen the policy — reduce entropy by concentrating probability mass on the mode of the output distribution.

The key insight: **the target matters more than the samples.** Experiments show that oversampling consistently outperforms non-oversampled training, while changing the composition of the training batch (e.g., balanced 4+4 from majority/rest clusters) does not improve performance.

## Why Training Toward the Densest Region Works

Press et al. (2024), "The Entropy Enigma," shows that entropy minimization has two phases:

- **Phase 1 (beneficial):** Pulls representations toward the nearest cluster center in embedding space. Representations become more organized and internally consistent. Accuracy improves as a side effect.
- **Phase 2 (harmful):** After many steps, representations overshoot the cluster center and drift away. The model becomes confidently wrong.

The critical point: Phase 1 works without labels. The mechanism is representation organization, not label correction. A model that consistently produces one answer (even if wrong) has more structured representations than a model that produces scattered answers. More structured representations generalize better to other prompts where the dense region IS correct.

## Oversampling as Mode Estimation

The M=32 → K=8 pipeline:

1. **M=32 generation:** Sample the current policy densely to estimate its mode
2. **Majority vote:** Identify the densest region (the mode)
3. **Binary reward:** Label each response as "in the mode" (1) or "not in the mode" (0)
4. **K=8 selection:** Pick responses from the mode to train on
5. **GRPO update:** Push the policy to concentrate more mass on the mode

Steps 1-2 are a **density estimation** problem. M=32 gives a lower-variance estimate of the true mode than M=8. Steps 3-5 use that estimate to sharpen the policy.

The 75% of responses discarded after selection are not wasted — they contributed to the density estimate in step 2. The compute cost of generating them is the cost of a better compass.

### Statistical Intuition

Think of it as kernel density estimation from samples:

- **M=8:** Sparse sampling. The estimated mode can jump between steps (high variance estimator). The gradient direction fluctuates — sometimes pointing toward the true mode, sometimes toward a spurious local density peak.
- **M=32:** Denser sampling. The estimated mode is stable and closer to the true mode (low variance estimator). The gradient direction is consistent across steps.

Consistent gradient direction → less random drift → the policy moves more efficiently toward the distribution center.

## Why the Target Matters More Than the Samples

### Evidence: Balanced 4+4 Selection

We tested selecting 4 responses from the majority cluster and 4 from the rest cluster, instead of 8 from the majority. This did not improve performance.

### Interpretation

- The 4 rest-cluster responses are from sparse regions of the distribution
- Training on sparse-region responses with negative advantages pushes the model away from those regions — but the model was already unlikely to go there
- The gradient from "move away from sparse regions" is redundant with "move toward dense regions"
- The rest-cluster responses dilute the batch with samples that carry less useful gradient signal
- The target (majority vote from M=32) is the same either way — the direction is right, but the 4+4 batch is a less efficient way to move in that direction

This confirms: the value of oversampling is in the **target quality** (accurate mode estimation from M=32), not in the **sample composition** of the training batch.

## Connection to Distribution Drift

### Without Oversampling (M=K=8)

- Mode estimated from 8 samples — high variance
- The estimated mode can differ significantly between steps for the same prompt
- The model receives flickering gradient signals: "move this way" one step, "move that way" next step
- Flickering gradients cause random walk behavior in the policy — more drift from the original center
- Empirically: exp1 (M=K=8) shows volatile val scores (0.414 → 0.591 → 0.544), consistent with noisy gradient direction

### With Oversampling (M=32, K=8)

- Mode estimated from 32 samples — low variance
- The estimated mode is stable across steps
- The model receives consistent gradient signals pointing toward the same region
- Consistent gradients cause directed movement — less drift, more efficient convergence
- Empirically: oversampled runs show more stable performance trajectories across multiple trials

### Extending Phase 1

Oversampling extends the beneficial Phase 1 window because denser sampling (M=32) provides a more robust mode estimate that resists degeneracy longer than M=8. Even as the policy sharpens and entropy drops, 32 samples still capture more of the mode's shape than 8 samples. The density estimate remains informative for longer before collapsing to a single point.

## Compute Trade-off

| Metric | M=K=8 | M=32, K=8 |
|--------|-------|-----------|
| Responses generated per step | 512 | 2048 |
| Responses trained on per step | 512 | ~520 |
| Tokens processed per step | ~220K | ~780K |
| Generation time per step | ~62s | ~182s |
| Total step time | ~472s | ~524s |

The 3.5x token generation overhead is the cost of a better compass. A better compass means fewer wasted gradient steps — the policy moves more consistently toward the true mode instead of random-walking due to noisy mode estimates. In practice, oversampled runs reach stable performance levels more reliably, even if the per-step cost is higher.

## Implications for Weighting Design

The KL-based drift weighting (Approach C from `docs/embedding_drift_weighting_plan.md`) complements oversampling:

- **Oversampling** improves the target: better mode estimation → better gradient direction
- **KL-based weighting** improves the step size per response: how much to trust each response's gradient based on its drift from the pretrained distribution

Together: better direction + better per-response calibration. The KL signal can detect when a selected response, despite agreeing with the majority vote, has drifted far from the pretrained distribution — meaning it might be "in the mode for the wrong reasons." Downweighting such responses further stabilizes training.

## Summary

Oversampling is fundamentally sound because:

1. It is a variance reduction technique for mode estimation, not a sample selection technique
2. Better mode estimation → more consistent gradient direction → less drift
3. Training toward the densest region works regardless of correctness (Press et al. Phase 1)
4. The compute overhead buys a better compass; fewer wasted gradient steps compensate for the cost
5. It extends the Phase 1 window by providing more robust density estimates as the policy sharpens
6. Empirical evidence confirms: the target (mode estimate quality) matters more than the sample composition
