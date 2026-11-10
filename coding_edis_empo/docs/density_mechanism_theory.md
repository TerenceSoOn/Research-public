# Density-Driven Representation Organization in Unsupervised RL

## Thesis

The mechanism of unsupervised RL (EMPO and similar) is not reward-driven learning but **density-driven representation organization**. The model's representations are pushed toward existing dense areas of the output distribution. Accuracy improves as a side effect because dense areas in a well-pretrained model are close to correct answers — pretraining already organized representations around truth.

This reframes EMPO's components:
- **Majority voting** is not "finding the right answer" — it is **density estimation** (identifying the mode)
- **Oversampling (M>K)** is not "better sample selection" — it is **variance reduction for density estimation**
- **Entropy minimization** is not the objective — it is a **consequence** of representation organization
- **Reward** is not the learning signal — it is a **proxy for density** that creates gradients pointing toward the mode

## What Zhang et al. (2025) Established

The EMPO paper shows:
- Unsupervised RL (no labels) achieves comparable performance to supervised RL
- The mechanism works through semantic entropy minimization
- Pass@k analysis reveals RL doesn't create new reasoning — it improves sampling efficiency from existing capabilities
- They conclude: "RL post-training appears to primarily refine the model's ability to efficiently access, prioritize, and consistently select latent reasoning patterns"

## What They Did NOT Establish

1. They did not separate the **target** (what to train toward) from the **reward** (how to assign gradient)
2. They did not test whether training toward non-majority clusters (2nd densest) also works
3. They did not connect their findings to the representation drift literature (Press et al. 2024)
4. They did not explain **why** consistency correlates with correctness — they observe it but attribute it to "well-pre-trained models"
5. They mix density estimation, sample selection, and reward design into one system — no ablation separates these factors

## What Shao et al. (2025) "Spurious Rewards" Established

Shao et al. show that RLVR with spurious rewards (random, incorrect, format-only) can still improve math reasoning on certain models:

- **Random rewards** (γ=0.5, completely uninformative) improve Qwen2.5-Math-7B on MATH-500 by +21.4%, nearly matching ground truth rewards (+29.1%)
- **Incorrect rewards** (rewarding only wrong answers via majority vote) still improve by +16.4%
- **The effect is model-family dependent:** works on Qwen2.5 models but NOT on OLMo or Llama families
- **GRPO's clipping is necessary:** without clipping, random rewards produce zero expected gradient. With clipping, asymmetric clipping ranges for high-probability vs low-probability tokens create a net positive gradient for already-probable tokens
- **The mechanism amplifies pre-existing behaviors:** for Qwen2.5-Math, RLVR amplifies code-based reasoning already present in pretraining

## What Shao et al. Did NOT Explain

1. They identify clipping bias and pre-existing behaviors as two separate factors but **don't connect them into a unified mechanism**
2. They observe that spurious rewards work on Qwen2.5 but not other families, and conjecture it reflects "shared pretraining distributions" — but **don't explain why pretraining distribution matters**
3. Their clipping bias analysis (Section 4) is purely about token-level probability, disconnected from their pre-existing behavior analysis (Section 5) — **no framework links the two**
4. They don't explain why incorrect rewards (rewarding wrong answers) still improve the model — this is deeply puzzling under any reward-centric framework
5. They frame the phenomenon as "spurious" and "counterintuitive" — suggesting they lack a framework that makes it intuitive

## Our Contribution

### 1. The Mechanistic "Why"

The field is stuck at "what works" without understanding "why it works." EMPO, GRPO, DAPO, and other unsupervised/semi-supervised RL methods are empirical recipes with post-hoc explanations. Zhang et al.'s conclusion — "RL refines the model's ability to select latent reasoning patterns" — is descriptive, not mechanistic.

We provide the mechanism: pretraining creates a representation landscape where dense areas correspond to correct reasoning. Density-driven training organizes representations toward these areas (Press et al. Phase 1). Correctness is a side effect of representation organization, not the objective.

### 2. Two-Level Density Framework

Zhang et al. treat EMPO as one system. We decompose it into two interacting levels of density:

**Token-Level Density (Autoregressive Self-Supervision)**

Autoregressive generation is inherently density-seeking. Every token in a sampled response was probable under the current policy — it was sampled from the model's own distribution. When you train on any sampled sequence (regardless of response-level reward), you are reinforcing token transitions that were already in the dense region of the token-level distribution.

This means: **any RL training on the model's own outputs is a form of token-level density training.** Even training toward a random cluster still pushes the model toward token sequences it already tends to generate. The token-level optimization is always self-supervised.

**Response-Level Density (Semantic Clustering)**

Majority voting adds a second layer of density selection on top of token-level density. Among all the token-level-probable sequences, it identifies which ones converge to the same semantic answer. This is density estimation in semantic space.

The two levels interact:
- Token-level density provides the base signal — any sampled sequence reinforces existing patterns
- Response-level density provides directional consistency — majority voting ensures the gradient points toward the same semantic region across steps
- Without response-level selection (random cluster), the token-level signal still works, but the gradient direction fluctuates between steps, causing less efficient convergence
- With response-level selection (majority cluster), the gradient direction is consistent, amplifying the token-level signal toward a specific semantic mode

This explains something existing work cannot: why even weak or noisy reward signals still improve models — the token-level mechanism is always working.

### 3. Factor Decomposition

Zhang et al. mix target, samples, and reward into one system. We separate them and show which factor actually drives improvement. Their ablations compare reward types (entropy vs. voting vs. gt) but don't test whether the target cluster matters independently of the reward.

### 4. The 2nd Cluster Experiment

The key empirical contribution. Training toward the 2nd densest cluster (excluding the majority) should still improve the model. This directly proves the mechanism is density, not correctness — something the existing framework can't explain because it frames everything as "finding the right answer through consensus."

### 5. Broad Explanatory Power

The framework explains with one unified mechanism:
- Why unsupervised RL works at all (token-level density is always self-supervised)
- Why majority voting helps (adds response-level directional consistency)
- Why oversampling helps (better density estimation, see `docs/oversampling_analysis.md`)
- Why RL eventually plateaus (Phase 2 overshooting from Press et al.)
- Why even noisy rewards still improve models (token-level mechanism is robust)
- Why the entropy-based advantage weighting degrades (see `docs/entropy_weighting_analysis.md`)

### 6. Unifying Shao et al.'s "Spurious Rewards" Findings

Shao et al. (2025) identify two separate factors — clipping bias and pre-existing behaviors — but don't connect them. Our framework unifies their findings:

**Their clipping bias = our token-level density mechanism.** GRPO's clipping creates asymmetric update ranges: high-probability tokens (dense region) can decrease more than they can increase, while low-probability tokens (sparse region) can increase more than they can decrease. Under random rewards, the net effect is that high-probability tokens get reinforced — the model moves toward its own dense areas. This is exactly token-level density training, and it works regardless of the reward signal.

**Their "pre-existing behaviors" = our pretraining density landscape.** Qwen2.5-Math has a strong code-reasoning mode in its pretraining distribution (a dense area). The clipping bias amplifies this existing dense area. OLMo and Llama lack this dense code-reasoning mode, so the clipping bias has no coherent direction to amplify — it pushes toward scattered, inconsistent dense areas, producing no net improvement.

**Why incorrect rewards still work:** Rewarding wrong answers still selects responses that were sampled from the current policy (token-level dense). The "incorrect" label at the response level is fighting the response-level density, but the token-level density mechanism still operates through clipping. The net effect is weaker than correct rewards (+16.4% vs +29.1%) because the two levels partially conflict, but the token-level mechanism still dominates.

**Why the effect is model-family dependent:** Models with a single dominant dense mode (Qwen2.5-Math's code reasoning) benefit from any training signal that amplifies that mode. Models without a dominant mode (OLMo, Llama) need response-level density guidance (correct rewards) to identify which sparse area to amplify. This predicts: the more concentrated a model's pretraining distribution, the less it needs informative rewards.

**What Shao et al. call "counterintuitive" becomes intuitive under our framework:** rewards are not the learning signal — density is. Rewards merely modulate the direction and efficiency of density-driven organization. When the pretraining landscape already has a strong mode, even random modulation amplifies it.

### Explaining the Performance Ordering: GT > Majority > Incorrect > Random

Shao et al. observe that ground-truth rewards outperform majority voting, and incorrect rewards outperform random rewards. Both findings are natural under our framework:

**Ground-truth > majority voting:**

Both are density-driven, but they differ in target stability. Majority voting estimates the mode of the *current* policy's output distribution — a moving target that co-evolves with the policy. As the policy sharpens, the estimated mode can overshoot or lock onto a slightly wrong region. Ground-truth rewards provide an external anchor: they always point toward the correct answer, which is near the pretrained mode but doesn't drift with the policy.

- Majority voting = density estimation of a moving target (current policy mode)
- Ground-truth = density estimation of a fixed target (correct answer, near the pretrained mode)

The fixed target produces more consistent gradients across steps → less drift → better final performance. This is fully consistent with our framework: both are density-driven, but ground-truth provides a more stable density signal.

**Incorrect label > random:**

"Incorrect" in Shao et al. means: identify the majority cluster via majority voting, then reward only responses that disagree with it. "Random" means: reward with probability γ=0.5 regardless of content.

Incorrect rewards still carry structural information about the output distribution. They identify the majority cluster and select the complement — the non-majority responses. These non-majority responses are still coherent clusters (2nd largest, 3rd largest, etc.), not random noise. Training toward them is training toward *a* dense area, just not the densest one. This is exactly our Exp-T2 scenario.

Random rewards carry zero structural information — they select responses uniformly, mixing dense and sparse regions. The only training signal comes from the token-level clipping bias, with no response-level directional consistency.

So the full ordering maps directly to density signal quality:

| Reward type | Response-level density signal | Token-level density signal | Predicted rank |
|-------------|------------------------------|---------------------------|----------------|
| Ground-truth | Fixed anchor near pretrained mode | Yes (clipping) | 1st (best) |
| Majority vote | Current policy mode (drifts) | Yes (clipping) | 2nd |
| Incorrect label | Non-majority clusters (still coherent dense areas) | Yes (clipping) | 3rd |
| Random | None (uniform selection) | Yes (clipping only) | 4th (worst) |

Each step down the ranking removes one layer of density information. The token-level mechanism (clipping bias) is always present, which is why even random rewards produce some improvement. The response-level signal adds directional consistency, and its quality determines the gap between reward types.

## The Full Mechanism

1. **Pretraining creates a representation landscape** where dense areas correspond to coherent, often correct reasoning patterns
2. **Token-level autoregressive sampling** always produces sequences from dense regions of the token distribution
3. **Response-level density estimation** (majority voting) identifies which semantic region is densest among the sampled responses
4. **Training toward dense areas organizes representations** — this is Press et al.'s Phase 1: pulling representations toward the nearest cluster center
5. **Organized representations generalize** — a model that consistently produces one answer has more structured internal representations, which transfer to other prompts
6. **Accuracy is a side effect** of representation organization, not the direct objective

The key prediction: **training toward ANY sufficiently dense area should improve the model**, even if that area is not the "correct" answer. The 2nd densest cluster should also provide useful training signal. Even random cluster selection should improve the model (due to token-level density), but less efficiently than density-guided selection.

## Connection to Prior Work

### Press et al. (2024) — "The Entropy Enigma"

Press et al. show entropy minimization works in Phase 1 by pulling representations toward the nearest cluster center. Our framework extends this in two ways:

1. **Two-level density:** Press et al. study a single level of representation organization. We identify two interacting levels — token-level autoregressive density (always present) and response-level semantic density (added by majority voting). The token-level mechanism explains why even random cluster selection improves the model; the response-level mechanism explains why majority voting improves it more.

2. **Density, not correctness:** EMPO's majority voting identifies the cluster center (density estimation). Training toward that center is Phase 1 of entropy minimization. The mechanism works for ANY dense cluster center, not just the "correct" one. Phase 2 failure (overshooting) explains why unsupervised RL eventually plateaus. Oversampling extends Phase 1 by providing more stable center estimates.

### Shao et al. (2025) — "Spurious Rewards"

Shao et al. provide strong empirical evidence for our theory without realizing it:

1. **Random rewards working = token-level density mechanism.** Their finding that random rewards improve Qwen2.5-Math-7B by +21.4% is predicted by our framework: the token-level density mechanism (amplified by GRPO clipping) operates regardless of the reward signal.

2. **Model-family dependence = pretraining density landscape.** Their observation that spurious rewards work on Qwen2.5 but not OLMo/Llama is explained by our framework: Qwen2.5-Math has a coherent dense mode (code reasoning) that the clipping bias can amplify. Other models lack this coherent mode.

3. **Incorrect rewards working = density > correctness.** Their most puzzling finding — that rewarding wrong answers still improves the model — is natural under our framework: the response-level reward is secondary to the token-level density mechanism. Even with inverted labels, the clipping bias still pushes toward high-probability tokens.

4. **Clipping as the enabler = density amplification mechanism.** Their ablation showing that removing clipping eliminates the random-reward effect confirms that the mechanism is density amplification (clipping biases toward high-probability tokens), not reward-driven learning.

Our framework provides the unified explanation they lack: clipping bias, pre-existing behaviors, and model-family dependence are all consequences of density-driven representation organization at two levels.

## Experimental Design: Factor Decomposition

### Factor 1: Target (what to train toward)

- **Exp-T1:** Train toward largest cluster only (current EMPO baseline)
- **Exp-T2:** Exclude largest cluster, train toward 2nd largest (the key experiment)
- **Exp-T3:** Train toward all clusters, weighted by density (frequency reward)
- **Exp-T4:** Train toward random cluster (tests token-level density without response-level consistency)
- **Exp-T5:** Train toward smallest cluster (adversarial — actively pushes away from dense areas)

Exp-T2 is the critical experiment. If training toward the 2nd densest cluster improves the model (even if less than T1), it proves the mechanism is density, not correctness.

Exp-T4 (random) is NOT a negative control — due to the autoregressive nature, token-level optimization is still self-supervised density training. Random cluster selection should still improve the model, but less efficiently than density-guided selection (inconsistent gradient direction across steps).

Exp-T5 (smallest cluster) is the true negative control. Training toward the least dense region actively fights both token-level and response-level density. This should hurt performance or show minimal improvement.

### Factor 2: Density estimation quality (oversampling)

- **Exp-D1:** M=8, K=8 (low-quality density estimate)
- **Exp-D2:** M=32, K=8 (high-quality density estimate)
- **Exp-D3:** M=64, K=8 (very high-quality density estimate, if compute allows)

Cross with Factor 1 to test: does better density estimation help more when training toward the 2nd cluster?

### Factor 3: Reward type (how density is encoded in the gradient)

- **Exp-R1:** Binary reward (1/0 based on cluster membership) — current setup
- **Exp-R2:** Frequency reward (reward = cluster_size/M) — original EMPO
- **Exp-R3:** Uniform positive reward for selected cluster (reward = 1.0 for all selected, regardless of cluster size)

This separates "which direction to push" (target) from "how hard to push" (reward magnitude).

### Priority Order

1. First: Exp-T2 (2nd cluster) — the key differentiating experiment
2. Second: Exp-T4 (random cluster) — tests token-level density hypothesis
3. Third: Exp-T5 (smallest cluster) — true negative control
4. Fourth: Exp-T1 vs Exp-R1 vs Exp-R2 — does density-graded reward help over binary?
5. Fifth: Cross Factor 1 x Factor 2 — does oversampling help more for 2nd cluster?

## Implementation Approach

The key change is in the reward manager: instead of always selecting the largest cluster, add a config option to select the Nth largest cluster. The rest of the pipeline (GRPO, GSPO, training) stays the same.

- Modify `src/edis_reward_manager.py`: add `target_cluster` config (largest, second_largest, random, smallest, all_weighted)
- Create new experiment scripts based on `scripts/run_exp_1.sh` (the clean M=K=8 baseline)
- Use M=32 for density estimation experiments (based on `scripts/run_exp_2.sh`)

## Expected Findings

1. Training toward 2nd cluster improves model (less than 1st, but significantly above baseline) — proves response-level density mechanism
2. Training toward random cluster ALSO improves model (less than 1st or 2nd) — proves token-level autoregressive density is the base mechanism; response-level selection adds directional consistency, not the fundamental signal
3. Training toward smallest cluster does NOT improve model (or hurts) — proves that actively fighting density is harmful; this is the true negative control
4. The improvement ordering should be: T1 (largest) > T2 (2nd largest) > T4 (random) > baseline > T5 (smallest) — this gradient directly maps to the density of the target
5. Better density estimation (M=32) helps more for ambiguous targets (2nd cluster, random) — proves oversampling is about estimation quality
6. Frequency reward outperforms binary reward — proves graded density signal is better than binary

## Significance

This work provides the first mechanistic explanation for why unsupervised RL improves LLM reasoning. It unifies three lines of work that currently exist in isolation:

1. **Zhang et al. (2025) / EMPO:** demonstrates unsupervised RL works but attributes it to "refining sampling efficiency" without explaining why
2. **Shao et al. (2025) / Spurious Rewards:** demonstrates that even random/incorrect rewards improve certain models but calls it "counterintuitive" and explains it through two disconnected factors (clipping bias + pre-existing behaviors)
3. **Press et al. (2024) / Entropy Enigma:** explains the Phase 1/Phase 2 dynamics of entropy minimization through embedding drift but doesn't connect to RL training

Our density-driven framework provides a single mechanism that explains all three: RL training drives representations toward existing dense areas of the output distribution, operating at two levels (token-level via clipping, response-level via reward selection). Accuracy is a side effect of representation organization, not the objective. The framework makes falsifiable predictions (T1 > T2 > T4 > baseline > T5) that no existing theory predicts, and explains the model-family dependence observed by Shao et al. as a consequence of pretraining density landscape structure.

## Experimental Details

### Models

**Primary: Qwen3-4B-Base** (`runtime/pretrained/qwen3-4b-base`)
- Family: Qwen (Alibaba), general-purpose base model
- Current experimental model with existing baselines (exp1, exp3)
- ~8 min/step on 2x A100-80G
- Existing results can be reused as T1 baseline — no need to re-run

**Secondary: Llama-3.2-3B-Instruct** (`runtime/pretrained/Llama-3.2-3B-Instruct`)
- Family: Llama (Meta), instruction-tuned model
- Different architecture, pretraining data, tokenizer — strongest generality claim
- ~5-6 min/step estimated on 2x A100-80G
- Already available locally
- Tests whether the density mechanism holds across model families AND for instruction-tuned models (whose output distribution is already sharpened by RLHF/DPO). If the mechanism works on an instruct model, it proves the theory is robust to the starting distribution shape — not specific to base models or a particular pretraining recipe
- Run only key experiments (T1, T2, T5) for generality

### Datasets

**Training:** NM_20K_1010.parquet (~20K math problems, same as all existing experiments)

**Evaluation (primary):** amc23_1010.parquet — AMC 2023, metric: `val-core/amc/acc/mean@16`

**Evaluation (secondary):** math500.parquet, minerva_math.parquet — additional eval benchmarks for robustness. These are inference-only and cheap to run.

### GPU Resources

- 4x A100-80G with NVLink
- Run 2 experiments in parallel (2 GPUs each)
- All models (Qwen3-4B, Llama-3.2-3B) fit comfortably on 2x A100-80G

### Training Duration

From existing logs:
- Exp1 (M=K=8, no weighting): peaks at step 20, plateaus by step 40
- Exp3 (M=K=8, weighted): peaks at step 10, flat after step 20

**50 steps is sufficient.** The signal is clear within 20-30 steps. Eval every 10 steps gives 5 data points to track the trajectory. Total training steps set to 50 (down from the default 500) to minimize compute.

### Metrics to Track

For each experiment:
1. `val-core/amc/acc/mean@16` — primary accuracy metric (eval every 10 steps)
2. `actor/entropy` — entropy trajectory (should decrease for all density-aligned targets)
3. `critic/score/mean` — training reward trajectory
4. `actor/pg_loss` — policy gradient loss magnitude
5. `actor/pg_clipfrac` — how often GSPO clip is hit
6. Secondary eval on math500 and minerva_math at step 50

### Experiment Matrix

#### Phase 1: Core Ablation (must run)

All use Qwen3-4B-Base, M=K=8, binary reward, 50 steps, eval every 10 steps.

| Exp | Target | Status | Est. Time (2 GPU) |
|-----|--------|--------|-------------------|
| T1 | Largest cluster | Done (exp1) | 0h |
| T2 | 2nd largest cluster (exclude largest) | To run | ~7h |
| T4 | Random cluster | To run | ~7h |
| T5 | Smallest cluster | To run | ~7h |

T1 is already done (exp1 results). T2, T4, T5 need 3 runs. With 2 machines (2 experiments on machine B, 1 on machine A): **~14h wall time.**

#### Phase 2: Oversampling Cross (if Phase 1 confirms theory)

Tests whether better density estimation helps more for non-majority targets.

| Exp | Target | M | K | Status | Est. Time |
|-----|--------|---|---|--------|-----------|
| T1-D2 | Largest | 32 | 8 | Done (exp2) | 0h |
| T2-D2 | 2nd largest | 32 | 8 | To run | ~7.5h |

#### Phase 3: Reward Type (if compute allows)

| Exp | Target | Reward | Est. Time |
|-----|--------|--------|-----------|
| T1-R2 | Largest | Frequency (cluster_size/M) | ~7h |
| T2-R2 | 2nd largest | Frequency | ~7h |

#### Phase 4: Generality — Llama-3.2-3B-Instruct (different model family)

M=K=8, binary reward, 50 steps. Only key experiments.

| Exp | Target | Est. Time |
|-----|--------|-----------|
| T1-Llama | Largest | ~5h |
| T2-Llama | 2nd largest | ~5h |
| T5-Llama | Smallest | ~5h |

Tests the density mechanism on a different model family (Meta vs Alibaba) and a different model type (instruct vs base). If T1 > T2 > baseline > T5 holds for Llama-3.2-3B-Instruct as well, the theory generalizes beyond Qwen base models.

### Compute Budget Summary

| Phase | New Experiments | Wall Time (2 parallel on 4x A100-80G) |
|-------|----------------|---------------------------------------|
| Phase 1 (core) | T2, T4, T5 | ~14h (T2+T4 parallel, then T5) |
| Phase 2 (oversampling) | T2-D2 | ~7.5h |
| Phase 3 (reward type) | T1-R2, T2-R2 | ~7h (parallel) |
| Phase 4 (generality) | T1/T2/T5 on Llama | ~10h (T1+T2 parallel, then T5) |
| **Total** | **10 experiments** | **~38.5h** |

**Minimum viable (Phase 1 only): ~14 hours wall time, 3 new experiments.**

Phase 1 alone is sufficient for the core claim. Phases 2-4 strengthen it for publication.

### Implementation Changes

1. Add `target_cluster` config to `src/edis_reward_manager.py`: options `largest` (default), `second_largest`, `random`, `smallest`
2. Create experiment scripts from `scripts/run_exp_1.sh` as template (cleanest baseline config)
3. Each new script only changes the `target_cluster` parameter and `TOTAL_TRAINING_STEPS=50` — everything else identical to exp1
