# Offline Evaluation Workflow

This directory contains the offline checkpoint evaluator. It is separate from
the online training scripts: training still validates only on AMC23, while this
workflow evaluates saved checkpoints across math, multiple-choice, and coding
benchmarks.

## Entry Point

```bash
cd /home/pterence6699/coding_edis_empo
bash scripts/run_offline_eval_queue.sh
```

Edit `scripts/run_offline_eval_queue.sh` to choose datasets, checkpoints, GPUs,
and smoke/full settings. The script generates a resolved queue YAML internally
and writes results under `eval_runs/<run_id>/`.

By default, the workflow resolves non-Git assets from:

```text
/home/pterence6699/coding_edis_empo/runtime/
```

Use `RUNTIME_ROOT`, `TRAIN_VENV`, `MODEL_PATH`, or `DATA_ROOT` only when
intentionally overriding the canonical runtime layout.

For a tiny GPU smoke test before full evaluation, set `math_row_limit` and/or
`nl_row_limit` in the queue. These limits now resolve to fixed subset artifacts
under `evaluation/subsets/`, so the same seed/size pair is reused across eval
runs instead of creating run-local parquet files.

## Dataset Groups

- Math: AMC23, AIME24, MATH500, GSM8K.
- Natural language: fixed stratified MMLU-Pro subset capped at 300 questions,
  fixed GPQA-Diamond subset when a cap is configured.
- Coding: fixed LiveCodeBench `release_v6` subset through the official runner.

MMLU-Pro, capped Math500, capped GSM8K, capped GPQA-Diamond, and optional smoke
subsets for AMC23/AIME24 use fixed subset files in `evaluation/subsets/`.
GSM8K is materialized as `runtime/math_data/gsm8k.parquet` the first time the
subset is prepared, unless that eval-ready parquet already exists. If the server
cannot reach Hugging Face, place the raw GSM8K test split at
`runtime/math_data/gsm8k_test.parquet` or `runtime/math_data/gsm8k_test.jsonl`;
it must contain `question` and `answer` columns, with `answer` using the standard
`#### final_answer` marker. The raw LiveCodeBench parquet should live under
`runtime/livecodebench/`, with the old `evaluation/subsets/` location kept only
as a migration fallback.
The subset generator will not overwrite existing subset artifacts unless
`force_prepare_subsets: true` is set in the queue.

## LiveCodeBench

LiveCodeBench uses a separate runtime under `.venv_lcb` and the official repo
under `external/LiveCodeBench`. The workflow bootstraps both when coding eval is
enabled and they are missing. FSDP checkpoints are merged to Hugging Face format
before running the official code-generation evaluator.
