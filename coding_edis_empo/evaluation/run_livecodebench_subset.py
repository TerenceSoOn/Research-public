"""Run official LiveCodeBench on a fixed question-id subset.

This wrapper leaves LiveCodeBench scoring intact. It only filters the official
benchmark list before the official runner generates and evaluates outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_subset_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("question_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{path} does not contain a non-empty `question_ids` list")
    return {str(item) for item in ids}


def _inject_local_model(model_key: str, model_repr: str, model_style: str) -> None:
    from lcb_runner.lm_styles import LMStyle, LanguageModel, LanguageModelStore

    if model_key in LanguageModelStore:
        return
    try:
        style = LMStyle[model_style]
    except KeyError as exc:
        raise ValueError(f"Unknown LiveCodeBench LMStyle {model_style!r}") from exc
    LanguageModelStore[model_key] = LanguageModel(
        model_name=model_key,
        model_repr=model_repr,
        model_style=style,
        release_date=datetime(2024, 1, 1),
        link=None,
    )


def _extract_metric_value(metrics: Any, name: str) -> float | None:
    if isinstance(metrics, dict):
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        for child in metrics.values():
            found = _extract_metric_value(child, name)
            if found is not None:
                return found
    elif isinstance(metrics, list):
        for child in metrics:
            found = _extract_metric_value(child, name)
            if found is not None:
                return found
    return None


def _find_eval_files(work_dir: Path) -> list[Path]:
    output_dir = work_dir / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.rglob("*_eval.json"), key=lambda path: path.stat().st_mtime)


def _prepare_work_dir_for_lcb_import(work_dir: Path) -> None:
    """Expose LiveCodeBench prompt assets at the relative path it expects."""
    spec = importlib.util.find_spec("lcb_runner")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate `lcb_runner` on PYTHONPATH")

    lcb_package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    source_prompts = lcb_package_dir / "prompts"
    target_package_dir = work_dir / "lcb_runner"
    target_prompts = target_package_dir / "prompts"
    target_package_dir.mkdir(parents=True, exist_ok=True)

    if target_prompts.exists() or target_prompts.is_symlink():
        return
    try:
        target_prompts.symlink_to(source_prompts, target_is_directory=True)
    except OSError:
        shutil.copytree(source_prompts, target_prompts)


def _local_value_to_lcb_input(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            import math

            if math.isnan(value):
                return ""
        except Exception:
            pass
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    return value


def _load_local_code_generation_dataset(
    dataset_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Any]:
    import pandas as pd
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".jsonl":
        df = pd.read_json(dataset_path, lines=True)
    elif dataset_path.suffix == ".json":
        df = pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported LiveCodeBench local dataset format: {dataset_path}")

    required = {
        "question_title",
        "question_content",
        "platform",
        "question_id",
        "contest_id",
        "contest_date",
        "starter_code",
        "difficulty",
        "public_test_cases",
        "private_test_cases",
        "metadata",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LiveCodeBench local dataset is missing columns: {missing}")

    records = []
    for row in df.to_dict(orient="records"):
        payload = {key: _local_value_to_lcb_input(row[key]) for key in required}
        records.append(CodeGenerationProblem(**payload))

    if start_date is not None:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        records = [item for item in records if start_dt <= item.contest_date]
    if end_date is not None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        records = [item for item in records if item.contest_date <= end_dt]

    print(f"[LCB] Loaded {len(records)} local code-generation problems from {dataset_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiveCodeBench on a fixed subset.")
    parser.add_argument("--subset-ids", type=Path, required=True)
    parser.add_argument("--local-model-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--local-dataset-path", type=Path)
    parser.add_argument("--model-key", default="local-qwen2.5-math")
    parser.add_argument("--model-repr", default="Local-Qwen2.5-Math")
    parser.add_argument("--model-style", default="CodeQwenInstruct")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    subset_ids = _load_subset_ids(args.subset_ids)
    if not args.local_model_path.exists():
        raise FileNotFoundError(f"Merged HF model path not found: {args.local_model_path}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    _prepare_work_dir_for_lcb_import(args.work_dir)
    os.chdir(args.work_dir)
    try:
        import lcb_runner.runner.main as lcb_main
    except Exception as exc:
        os.chdir(old_cwd)
        raise RuntimeError(
            "LiveCodeBench is not importable. Run through scripts/run_offline_eval_queue.sh "
            "so the isolated .venv_lcb environment is bootstrapped first."
        ) from exc

    _inject_local_model(args.model_key, args.model_repr, args.model_style)

    original_build_prompt_benchmark = lcb_main.build_prompt_benchmark

    def build_filtered_prompt_benchmark(lcb_args):
        if args.local_dataset_path is not None:
            from lcb_runner.prompts import format_prompt_generation
            from lcb_runner.utils.scenarios import Scenario

            if lcb_args.scenario != Scenario.codegeneration:
                raise ValueError("Local LiveCodeBench dataset support is codegeneration-only")
            benchmark = _load_local_code_generation_dataset(
                args.local_dataset_path,
                start_date=lcb_args.start_date,
                end_date=lcb_args.end_date,
            )
            benchmark = sorted(benchmark, key=lambda item: item.question_id)
            format_prompt = format_prompt_generation
        else:
            benchmark, format_prompt = original_build_prompt_benchmark(lcb_args)
        before = len(benchmark)
        filtered = [item for item in benchmark if str(item.question_id) in subset_ids]
        found_ids = {str(item.question_id) for item in filtered}
        missing = sorted(subset_ids - found_ids)
        if missing:
            raise RuntimeError(
                f"LiveCodeBench subset contains {len(missing)} ids not present in "
                f"{lcb_args.release_version}: {missing[:10]}"
            )
        print(f"[LCB] Filtered benchmark from {before} to {len(filtered)} questions.")
        return filtered, format_prompt

    lcb_main.build_prompt_benchmark = build_filtered_prompt_benchmark

    sys.argv = [
        "lcb_runner.runner.main",
        "--model",
        args.model_key,
        "--local_model_path",
        str(args.local_model_path),
        "--trust_remote_code",
        "--scenario",
        "codegeneration",
        "--evaluate",
        "--release_version",
        args.release_version,
        "--n",
        str(args.n),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--num_process_evaluate",
        str(args.num_process_evaluate),
        "--timeout",
        str(args.timeout),
        "--max_tokens",
        str(args.max_tokens),
        "--dtype",
        args.dtype,
        "--use_cache",
    ]
    if args.debug:
        sys.argv.append("--debug")

    before_files = set(_find_eval_files(args.work_dir))
    try:
        lcb_main.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    after_files = _find_eval_files(args.work_dir)
    new_files = [path for path in after_files if path not in before_files]
    eval_file = new_files[-1] if new_files else (after_files[-1] if after_files else None)
    if eval_file is None:
        raise RuntimeError(f"LiveCodeBench did not produce an *_eval.json under {args.work_dir}")

    metrics = json.loads(eval_file.read_text(encoding="utf-8"))
    eval_all_file = Path(str(eval_file).replace("_eval.json", "_eval_all.json"))
    payload = {
        "dataset": "livecodebench",
        "release_version": args.release_version,
        "subset_ids": str(args.subset_ids),
        "subset_size": len(subset_ids),
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pass@1": _extract_metric_value(metrics, "pass@1"),
        "pass@5": _extract_metric_value(metrics, "pass@5"),
        "raw_eval_file": str(eval_file),
        "raw_eval_all_file": str(eval_all_file) if eval_all_file.exists() else None,
        "raw_metrics": metrics,
    }
    args.metrics_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
