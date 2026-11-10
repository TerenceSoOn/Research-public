# Oversampled EMPO Experiments

This repo contains the configurable oversampling / selection layer used on top of
`verl-empo` for unsupervised math-reasoning RL experiments.

The active experiment surface is now EMPO-only:

- baseline EMPO on `N=K`
- oversampled EMPO with `M > K`
- random `K` selection from the oversampled pool
- balanced largest-vs-rest selection from the oversampled pool
- optional entropy-based post-GRPO advantage weighting

The configurable oversampled path never recomputes reward after selection. It:

1. runs EMPO once on the original `M` responses
2. selects `K` responses
3. trains those selected responses using reward reused from the first-round EMPO artifact

## Active Reward Modes

- `use_empo_reward`
  - selected samples inherit their original first-round EMPO reward
- `use_binary_reward`
  - selected samples inherit binary `1/0` reward from first-round winner support

## Active Selection Modes

- `random_topk`
  - sample `K` responses uniformly from the `M`-sample pool
- `balanced_largest_vs_rest`
  - sample `group_largest_count` responses from the first-round EMPO winner cluster
  - sample `group_rest_count` responses from all remaining responses
  - active balanced preset is method `D`: random from largest, random from rest

If either balanced pool is underfilled, the whole prompt is skipped.

## Experiment Scripts

- `scripts/run_A_empo_baseline.sh`
  - experiment A
  - original EMPO `entropy` reward on `K=8`

- `scripts/run_B_empo_oversampled_random.sh`
  - experiment B
  - `M=32`, random `K=8`, first-round binary `1/0` reward

- `scripts/run_C_empo_oversampled_methodD.sh`
  - experiment C
  - `M=32`, random 4 from largest cluster and random 4 from rest, first-round binary `1/0` reward

- `scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`
  - experiment D
  - same as C, plus `adv_weight_mode=entropy_high_token_ratio`

Compatibility scripts:

- `scripts/run_empo_baseline.sh`
  - legacy standalone baseline launcher
- `scripts/run_empo_oversampled_32.sh`
  - alias for experiment B
- `scripts/run_edis_filtering_32.sh`
  - legacy configurable oversampled/filtering launcher kept for compatibility

The A/B/C/D scripts are full standalone launchers. They do not `exec` or source
the legacy scripts above.

By default, A/B/C/D all log to the same W&B project/group:

- W&B project: `EMPO-Only-ABCD`
- W&B group: `EMPO-Only-ABCD`

The run names remain distinct through the experiment-specific `run_signature`.

## Entropy-Based Advantage Weighting

Experiment D keeps the same reward as experiment C and only changes the policy
update:

- compute token entropies over the current batch
- find the top-20% entropy threshold over valid tokens
- compute each response's fraction of high-entropy tokens
- convert that fraction into a bounded inverse-style advantage weight
- apply the weight after GRPO computes advantages

This is controlled by:

- `adv_weight_mode=off | entropy_high_token_ratio`
- `entropy_weight_top_fraction`
- `adv_weight_clip_min`
- `adv_weight_clip_max`

## Project Structure

```text
coding_edis_empo/
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── edis_reward_manager.py
│   └── config/
│       └── edis_empo_trainer.yaml
└── scripts/
    ├── run_A_empo_baseline.sh
    ├── run_B_empo_oversampled_random.sh
    ├── run_C_empo_oversampled_methodD.sh
    ├── run_D_empo_oversampled_methodD_entropy_adv.sh
    ├── run_empo_baseline.sh
    ├── run_empo_oversampled_32.sh
    └── run_edis_filtering_32.sh
```

Note: file/module names still contain `edis_*` for compatibility while the active
runtime behavior has been refactored to EMPO-only logic. Legacy EDIS-only code
can be removed in a later cleanup pass.
