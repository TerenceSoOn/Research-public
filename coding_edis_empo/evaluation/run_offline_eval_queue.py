"""Queue runner for offline multi-checkpoint evaluation."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml as _pyyaml
except Exception:
    _pyyaml = None

from evaluation.prepare_eval_subsets import (
    DEFAULT_DATA_ROOT,
    DEFAULT_LCB_LOCAL_DATASET,
    MAX_GSM8K_SUBSET_SIZE,
    MAX_GPQA_DIAMOND_SUBSET_SIZE,
    MAX_MATH500_SUBSET_SIZE,
    MAX_MMLU_PRO_SUBSET_SIZE,
    SEED,
    SUBSET_DIR,
    prepare_all,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("RUNTIME_ROOT", PROJECT_ROOT / "runtime")).expanduser()
DEFAULT_VERL_ROOT = PROJECT_ROOT.parent / "coding_empo" / "project_EMPO" / "EMPO" / "verl-empo"
DEFAULT_EMPO_ROOT = DEFAULT_VERL_ROOT.parent
DEFAULT_PROJECT_EMPO_ROOT = DEFAULT_VERL_ROOT.parents[1]


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MODEL_PATH = (
    Path(os.environ["MODEL_PATH"]).expanduser()
    if "MODEL_PATH" in os.environ
    else _first_existing_path(
        [
            DEFAULT_RUNTIME_ROOT / "pretrained" / "Qwen2.5-math-1.5B",
            Path("/home/pterence6699/coding_edis/pretrained/Qwen2.5-math-1.5B"),
        ]
    )
)
DEFAULT_TRAIN_DATA = _first_existing_path(
    [
        DEFAULT_DATA_ROOT / "NM_20K_1010.parquet",
        DEFAULT_EMPO_ROOT / "math_data" / "NM_20K_1010.parquet",
    ]
)


DATASETS = {
    "math": [
        ("amc23", "amc", "amc23_1010.parquet"),
        ("aime24", "AIME24", "aime24_0726.parquet"),
        ("math500", "math500", "math500.parquet"),
        ("gsm8k", "gsm8k", None),
    ],
    "nl": [
        ("mmlu_pro", "mmlu_pro", None),
        ("gpqa_diamond", "gpqa_diamond", "gpqa_diamond_eval.parquet"),
    ],
}


@dataclass
class EvalResult:
    checkpoint_id: str
    group: str
    dataset: str
    primary_metric: str
    primary_value: float | None
    secondary_metric: str | None
    secondary_value: float | None
    status: str
    metric_file: str
    log_file: str
    error: str = ""


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = _pyyaml.safe_load(text) if _pyyaml is not None else _simple_yaml_load(text)
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"Queue config must be a YAML mapping: {path}")
    return data


def _simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    return lines


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Small YAML subset parser for the queue file when PyYAML is unavailable."""
    lines = _simple_yaml_lines(text)
    result: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        indent, line = lines[i]
        if indent != 0:
            raise ValueError(f"Unsupported YAML indentation at line: {line}")
        if line.endswith(":"):
            key = line[:-1].strip()
            i += 1
            items: list[Any] = []
            while i < len(lines) and lines[i][0] > indent:
                child_indent, child_line = lines[i]
                if not child_line.startswith("- "):
                    raise ValueError(f"Expected YAML list item under {key!r}: {child_line}")
                rest = child_line[2:].strip()
                if ": " in rest:
                    item: dict[str, Any] = {}
                    sub_key, sub_value = rest.split(": ", 1)
                    item[sub_key.strip()] = _simple_yaml_scalar(sub_value)
                    i += 1
                    while i < len(lines) and lines[i][0] > child_indent:
                        _, nested_line = lines[i]
                        if ": " not in nested_line:
                            raise ValueError(f"Unsupported nested YAML line: {nested_line}")
                        sub_key, sub_value = nested_line.split(": ", 1)
                        item[sub_key.strip()] = _simple_yaml_scalar(sub_value)
                        i += 1
                    items.append(item)
                else:
                    items.append(_simple_yaml_scalar(rest))
                    i += 1
            result[key] = items
        elif ": " in line:
            key, value = line.split(": ", 1)
            result[key.strip()] = _simple_yaml_scalar(value)
            i += 1
        else:
            raise ValueError(f"Unsupported YAML line: {line}")
    return result


def _safe_dump_yaml(payload: dict[str, Any]) -> str:
    if _pyyaml is not None:
        return _pyyaml.safe_dump(payload, sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "checkpoint"


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"Expected list or comma-separated string, got {value!r}")


def _gpu_count(gpu_ids: str) -> int:
    return len([item for item in gpu_ids.split(",") if item.strip()])


def _checkpoint_kind(checkpoint: Path) -> str:
    """Return the checkpoint format used by the offline evaluator."""
    if (checkpoint / "actor").is_dir():
        return "fsdp"
    if (checkpoint / "config.json").is_file():
        return "hf"
    return "unknown"


def _quote_list(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{str(path)}'" for path in paths) + "]"


def _find_python(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "No usable Python executable found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _site_packages_for_python(python_bin: str) -> str:
    code = "import site; print(site.getsitepackages()[0])"
    return subprocess.check_output([python_bin, "-c", code], text=True).strip()


def _base_env(config: dict[str, Any], gpu_ids: str) -> dict[str, str]:
    verl_root = Path(config["verl_root"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env.setdefault("VLLM_USE_V1", "0")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{verl_root}:{env.get('PYTHONPATH', '')}"
    env["EVAL_MATH_PATH"] = str(Path(config["empo_root"]) / "eval_math")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("RAY_ADDRESS", "local")
    return env


def _resolve_config(queue_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    run_id = raw.get("run_id") or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    gpu_ids = str(raw.get("gpu_ids", "0,3"))
    verl_root = Path(raw.get("verl_root") or os.environ.get("VERL_EMPO_ROOT") or DEFAULT_VERL_ROOT).expanduser()
    empo_root = Path(raw.get("empo_root") or os.environ.get("EMPO_ROOT") or DEFAULT_EMPO_ROOT).expanduser()
    project_empo_root = Path(
        raw.get("project_empo_root")
        or os.environ.get("PROJECT_EMPO_ROOT")
        or DEFAULT_PROJECT_EMPO_ROOT
    ).expanduser()
    data_root = Path(raw.get("data_root") or DEFAULT_DATA_ROOT).expanduser()
    output_root = Path(raw.get("output_root") or PROJECT_ROOT / "eval_runs").expanduser()
    run_dir = output_root / run_id
    train_venv = Path(raw.get("train_venv") or os.environ.get("TRAIN_VENV") or "").expanduser()
    if raw.get("python_bin"):
        python_bin = Path(raw["python_bin"]).expanduser()
    elif train_venv and str(train_venv) != ".":
        python_bin = train_venv / "bin" / "python3"
    else:
        python_bin = _find_python(
            [
                project_empo_root / "venv" / "bin" / "python3",
                project_empo_root.parent / "venv" / "bin" / "python3",
                Path("/home/pterence6699/coding_empo/venv/bin/python3"),
                PROJECT_ROOT / ".venv" / "bin" / "python3",
                Path(sys.executable),
            ]
        )

    resolved = {
        **raw,
        "queue_path": str(queue_path),
        "run_id": run_id,
        "gpu_ids": gpu_ids,
        "n_gpus": _gpu_count(gpu_ids),
        "verl_root": str(verl_root),
        "empo_root": str(empo_root),
        "project_empo_root": str(project_empo_root),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "python_bin": str(python_bin),
        "model_path": str(Path(raw.get("model_path") or DEFAULT_MODEL_PATH).expanduser()),
        "train_data": str(Path(raw.get("train_data") or DEFAULT_TRAIN_DATA).expanduser()),
        "groups": _as_list(raw.get("groups"), ["math", "nl", "coding"]),
        "math_n": int(raw.get("math_n", 16)),
        "nl_n": int(raw.get("nl_n", 16)),
        "livecodebench_n": int(raw.get("livecodebench_n", 10)),
        "val_do_sample": raw.get("val_do_sample", True),
        "val_temperature": float(raw.get("val_temperature", 0.2)),
        "val_top_p": float(raw.get("val_top_p", 0.95)),
        "val_top_k": int(raw.get("val_top_k", 20)),
        "livecodebench_temperature": float(raw.get("livecodebench_temperature", 0.2)),
        "livecodebench_top_p": float(raw.get("livecodebench_top_p", 0.95)),
        "livecodebench_release": raw.get("livecodebench_release", "release_v6"),
        "livecodebench_subset_size": int(raw.get("livecodebench_subset_size", 200)),
        "math500_subset_size": min(
            int(raw.get("math500_subset_size", 100)),
            MAX_MATH500_SUBSET_SIZE,
        ),
        "gsm8k_subset_size": min(
            int(raw.get("gsm8k_subset_size", 100)),
            MAX_GSM8K_SUBSET_SIZE,
        ),
        "gpqa_subset_size": min(
            int(raw.get("gpqa_subset_size", 100)),
            MAX_GPQA_DIAMOND_SUBSET_SIZE,
        ),
        "livecodebench_local_dataset_path": str(
            Path(raw.get("livecodebench_local_dataset_path") or DEFAULT_LCB_LOCAL_DATASET).expanduser()
        ),
        "mmlu_subset_size": min(
            int(raw.get("mmlu_subset_size", MAX_MMLU_PRO_SUBSET_SIZE)),
            MAX_MMLU_PRO_SUBSET_SIZE,
        ),
        "math_row_limit": raw.get("math_row_limit", None),
        "nl_row_limit": raw.get("nl_row_limit", None),
        # Keep dataset subset selection independent from generation sampling.
        # `seed` remains a backwards-compatible alias for subset_seed.
        "subset_seed": int(raw.get("subset_seed", raw.get("seed", SEED))),
        "eval_seed": int(raw.get("eval_seed", raw.get("seed", 0))),
        "fail_fast": bool(raw.get("fail_fast", False)),
        "force_prepare_subsets": bool(raw.get("force_prepare_subsets", False)),
    }
    return resolved


def _min_optional_limit(*limits: Any) -> Any:
    normalized: list[int] = []
    for value in limits:
        if value in (None, "", "null", "None"):
            continue
        normalized.append(int(value))
    if not normalized:
        return None
    return min(normalized)


def _hydra_optional_int(value: Any) -> str:
    if value in (None, "", "null", "None"):
        return "null"
    return str(int(value))


def _build_hydra_args(
    config: dict[str, Any],
    checkpoint: Path,
    val_files: list[Path],
    val_n: int,
    metrics_output: Path,
    hydra_dir: Path,
) -> list[str]:
    max_prompt_length = int(config.get("max_prompt_length", 2048))
    max_response_length = int(config.get("max_response_length", 3072))
    val_batch_size = _hydra_optional_int(config.get("val_batch_size", None))
    gpu_memory_utilization = float(config.get("gpu_memory_utilization", 0.6))
    val_do_sample = str(config.get("val_do_sample", True))
    val_temperature = float(config.get("val_temperature", 0.2))
    val_top_p = float(config.get("val_top_p", 0.95))
    val_top_k = int(config.get("val_top_k", 20))
    checkpoint_kind = _checkpoint_kind(checkpoint)
    model_path = checkpoint if checkpoint_kind == "hf" else Path(config["model_path"])

    args = [
        f"++data.train_files={_quote_list([Path(config['train_data'])])}",
        f"++data.val_files={_quote_list(val_files)}",
        "++data.use_shm=False",
        "++data.prompt_key=prompt",
        "++data.reward_fn_key=data_source",
        "++data.truncation='left'",
        f"++data.max_prompt_length={max_prompt_length}",
        f"++data.max_response_length={max_response_length}",
        "++data.train_batch_size=8",
        "++data.gen_batch_size=8",
        f"++data.val_batch_size={val_batch_size}",
        "++data.filter_overlong_prompts=False",
        "++data.validation_shuffle=False",
        f"++actor_rollout_ref.rollout.n={val_n}",
        "++algorithm.adv_estimator=grpo",
        "++algorithm.use_kl_in_reward=False",
        "++actor_rollout_ref.model.use_remove_padding=True",
        "++actor_rollout_ref.model.use_fused_kernels=False",
        "++actor_rollout_ref.actor.use_dynamic_bsz=True",
        "++actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"++actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_prompt_length + max_response_length}",
        f"++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
        f"++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_prompt_length + max_response_length}",
        f"++actor_rollout_ref.model.path={str(model_path)}",
        "++actor_rollout_ref.model.override_config.attention_dropout=0.",
        "++actor_rollout_ref.model.override_config.embd_pdrop=0.",
        "++actor_rollout_ref.model.override_config.resid_pdrop=0.",
        "++actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "++actor_rollout_ref.model.use_liger=False",
        "++actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "++actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "++actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
        "++actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"++actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        "++actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "++actor_rollout_ref.rollout.expert_parallel=False",
        "++actor_rollout_ref.rollout.enable_chunked_prefill=True",
        f"++actor_rollout_ref.rollout.max_num_batched_tokens={max_prompt_length + max_response_length}",
        f"++actor_rollout_ref.rollout.seed={int(config['eval_seed'])}",
        f"++actor_rollout_ref.rollout.val_kwargs.do_sample={val_do_sample}",
        f"++actor_rollout_ref.rollout.val_kwargs.n={val_n}",
        f"++actor_rollout_ref.rollout.val_kwargs.top_k={val_top_k}",
        f"++actor_rollout_ref.rollout.val_kwargs.top_p={val_top_p}",
        f"++actor_rollout_ref.rollout.val_kwargs.temperature={val_temperature}",
        "++actor_rollout_ref.rollout.enforce_eager=False",
        "++actor_rollout_ref.rollout.free_cache_engine=False",
        "++reward_model.reward_manager=empo",
        "++reward_model.reward_kwargs.reward_type=entropy",
        f"++reward_model.reward_kwargs.n_samples_per_prompt={val_n}",
        f"++reward_model.reward_kwargs.n_votes_per_prompt={val_n}",
        "++reward_model.use_xverify=False",
        "++reward_model.use_eval_math=True",
        "++trainer.logger=['console']",
        "++trainer.project_name=EMPO-Offline-Eval",
        "++trainer.experiment_name=offline_eval",
        f"++trainer.n_gpus_per_node={config['n_gpus']}",
        "++trainer.nnodes=1",
        "++trainer.val_before_train=True",
        "++trainer.val_only=True",
        "++trainer.test_freq=1",
        "++trainer.save_freq=-1",
        "++trainer.topk_checkpoint_enable=False",
        "++trainer.total_epochs=1",
        "++trainer.total_training_steps=1",
        f"++trainer.default_local_dir={str(hydra_dir / 'checkpoints')}",
        f"++trainer.validation_metrics_output={str(metrics_output)}",
        "++trainer.log_val_generations=0",
        f"hydra.run.dir={str(hydra_dir / 'hydra')}",
        f"hydra.searchpath=[file://{config['verl_root']}/verl/trainer/config]",
    ]
    return args


def _run_command(cmd: list[str], log_file: Path, env: dict[str, str], cwd: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        log.write("[Command]\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode


def _maybe_limited_fixed_subset(
    original_path: Path,
    subset_outputs: dict[str, str],
    subset_key: str,
    limit: Any,
) -> Path:
    if limit in (None, "", "null", "None"):
        return original_path
    limit_n = int(limit)
    if limit_n <= 0:
        return original_path
    if subset_key not in subset_outputs:
        raise KeyError(f"Fixed subset output missing for {subset_key}")
    return Path(subset_outputs[subset_key])


def _find_metric(metrics: dict[str, Any], data_source: str, var: str, metric_prefix: str, n: int) -> tuple[str | None, float | None]:
    exact = f"val-core/{data_source}/{var}/{metric_prefix}@{n}"
    if exact in metrics and isinstance(metrics[exact], (int, float)):
        return exact, float(metrics[exact])
    prefix = f"val-core/{data_source}/{var}/{metric_prefix}@"
    for key in sorted(metrics):
        if key.startswith(prefix) and isinstance(metrics[key], (int, float)):
            return key, float(metrics[key])
    return None, None


def _parse_verl_metrics(
    checkpoint_id: str,
    group: str,
    metric_file: Path,
    log_file: Path,
    val_n: int,
) -> list[EvalResult]:
    payload = json.loads(metric_file.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    rows: list[EvalResult] = []
    for dataset_name, data_source, _ in DATASETS[group]:
        primary_metric, primary_value = _find_metric(metrics, data_source, "acc", "mean", val_n)
        secondary_metric, secondary_value = _find_metric(metrics, data_source, "acc", "maj", val_n)
        if primary_metric is None:
            primary_metric, primary_value = _find_metric(metrics, data_source, "reward", "mean", val_n)
        rows.append(
            EvalResult(
                checkpoint_id=checkpoint_id,
                group=group,
                dataset=dataset_name,
                primary_metric=primary_metric or "",
                primary_value=primary_value,
                secondary_metric=secondary_metric,
                secondary_value=secondary_value,
                status="ok" if primary_value is not None else "missing_metric",
                metric_file=str(metric_file),
                log_file=str(log_file),
            )
        )
    return rows


def _run_verl_group(
    config: dict[str, Any],
    checkpoint_id: str,
    checkpoint: Path,
    group: str,
    val_files: list[Path],
    val_n: int,
    run_dir: Path,
) -> list[EvalResult]:
    checkpoint_for_eval = checkpoint
    if _checkpoint_kind(checkpoint) == "fsdp":
        checkpoint_for_eval = _merge_checkpoint_if_needed(config, checkpoint_id, checkpoint, run_dir)

    metric_file = run_dir / "raw_metrics" / checkpoint_id / f"{group}.json"
    log_file = run_dir / "logs" / checkpoint_id / f"{group}.log"
    hydra_dir = run_dir / "raw_metrics" / checkpoint_id / f"{group}_work"
    cmd = [
        config["python_bin"],
        "-m",
        "src.main",
        *_build_hydra_args(config, checkpoint_for_eval, val_files, val_n, metric_file, hydra_dir),
    ]
    code = _run_command(cmd, log_file, _base_env(config, config["gpu_ids"]), PROJECT_ROOT)
    if code != 0:
        return [
            EvalResult(
                checkpoint_id=checkpoint_id,
                group=group,
                dataset=dataset_name,
                primary_metric="",
                primary_value=None,
                secondary_metric=None,
                secondary_value=None,
                status="failed",
                metric_file=str(metric_file),
                log_file=str(log_file),
                error=f"Verl val-only command exited with {code}",
            )
            for dataset_name, _, _ in DATASETS[group]
        ]
    if not metric_file.exists():
        raise RuntimeError(f"Validation command succeeded but did not write {metric_file}")
    return _parse_verl_metrics(checkpoint_id, group, metric_file, log_file, val_n)


def _ensure_livecodebench_runtime(config: dict[str, Any], run_dir: Path) -> tuple[Path, Path]:
    external_root = Path(config.get("external_root") or PROJECT_ROOT / "external")
    lcb_root = Path(config.get("livecodebench_root") or external_root / "LiveCodeBench")
    lcb_venv = Path(config.get("livecodebench_venv") or PROJECT_ROOT / ".venv_lcb")
    bootstrap_log = run_dir / "logs" / "livecodebench_bootstrap.log"
    bootstrap_log.parent.mkdir(parents=True, exist_ok=True)

    with bootstrap_log.open("a", encoding="utf-8") as log:
        if not (lcb_root / "lcb_runner").exists():
            external_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/LiveCodeBench/LiveCodeBench.git", str(lcb_root)],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        if not (lcb_venv / "bin" / "python").exists():
            subprocess.run(
                [config["python_bin"], "-m", "venv", "--system-site-packages", str(lcb_venv)],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        subprocess.run(
            [str(lcb_venv / "bin" / "python"), "-m", "pip", "install", str(lcb_root)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    return lcb_root, lcb_venv / "bin" / "python"


def _merge_checkpoint_if_needed(
    config: dict[str, Any],
    checkpoint_id: str,
    checkpoint: Path,
    run_dir: Path,
) -> Path:
    if _checkpoint_kind(checkpoint) == "hf":
        return checkpoint

    target_dir = run_dir / "merged_models" / checkpoint_id
    if (target_dir / "config.json").exists():
        return target_dir

    log_file = run_dir / "logs" / checkpoint_id / "merge_model.log"
    cmd = [
        config["python_bin"],
        str(Path(config["verl_root"]) / "scripts" / "model_merger.py"),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint / "actor"),
        "--target_dir",
        str(target_dir),
    ]
    code = _run_command(cmd, log_file, _base_env(config, config["gpu_ids"]), Path(config["verl_root"]))
    if code != 0:
        raise RuntimeError(f"Model merge failed for {checkpoint_id}; see {log_file}")
    return target_dir


def _run_livecodebench(
    config: dict[str, Any],
    checkpoint_id: str,
    checkpoint: Path,
    subset_ids: Path,
    run_dir: Path,
) -> EvalResult:
    lcb_root, lcb_python = _ensure_livecodebench_runtime(config, run_dir)
    merged_model = _merge_checkpoint_if_needed(config, checkpoint_id, checkpoint, run_dir)

    metric_file = run_dir / "livecodebench" / checkpoint_id / "metrics.json"
    log_file = run_dir / "logs" / checkpoint_id / "livecodebench.log"
    work_dir = run_dir / "livecodebench" / checkpoint_id / "work"
    env = _base_env(config, config["gpu_ids"])
    env["VLLM_USE_V1"] = str(config.get("livecodebench_vllm_use_v1", "0"))
    train_site_packages = _site_packages_for_python(config["python_bin"])
    env["PYTHONPATH"] = (
        f"{lcb_root}:{PROJECT_ROOT}:{train_site_packages}:{env.get('PYTHONPATH', '')}"
    )
    cmd = [
        str(lcb_python),
        "-m",
        "evaluation.run_livecodebench_subset",
        "--subset-ids",
        str(subset_ids),
        "--local-model-path",
        str(merged_model),
        "--work-dir",
        str(work_dir),
        "--metrics-output",
        str(metric_file),
        "--release-version",
        str(config["livecodebench_release"]),
        "--n",
        str(config["livecodebench_n"]),
        "--temperature",
        str(config["livecodebench_temperature"]),
        "--top-p",
        str(config["livecodebench_top_p"]),
        "--tensor-parallel-size",
        str(config.get("livecodebench_tensor_parallel_size", config["n_gpus"])),
        "--num-process-evaluate",
        str(config.get("livecodebench_num_process_evaluate", 12)),
        "--timeout",
        str(config.get("livecodebench_timeout", 6)),
    ]
    lcb_local_dataset = Path(config["livecodebench_local_dataset_path"])
    if lcb_local_dataset.exists():
        cmd.extend(["--local-dataset-path", str(lcb_local_dataset)])
    if bool(config.get("livecodebench_debug", False)):
        cmd.append("--debug")

    code = _run_command(cmd, log_file, env, PROJECT_ROOT)
    if code != 0:
        return EvalResult(
            checkpoint_id=checkpoint_id,
            group="coding",
            dataset="livecodebench",
            primary_metric="pass@1",
            primary_value=None,
            secondary_metric="pass@5",
            secondary_value=None,
            status="failed",
            metric_file=str(metric_file),
            log_file=str(log_file),
            error=f"LiveCodeBench command exited with {code}",
        )
    payload = json.loads(metric_file.read_text(encoding="utf-8"))
    return EvalResult(
        checkpoint_id=checkpoint_id,
        group="coding",
        dataset="livecodebench",
        primary_metric="pass@1",
        primary_value=payload.get("pass@1"),
        secondary_metric="pass@5",
        secondary_value=payload.get("pass@5"),
        status="ok" if payload.get("pass@1") is not None else "missing_metric",
        metric_file=str(metric_file),
        log_file=str(log_file),
    )


def _write_summaries(run_dir: Path, rows: list[EvalResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "summary.csv"
    fieldnames = [
        "checkpoint_id",
        "group",
        "dataset",
        "primary_metric",
        "primary_value",
        "secondary_metric",
        "secondary_value",
        "status",
        "metric_file",
        "log_file",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    by_checkpoint: dict[str, dict[str, Any]] = {}
    for row in rows:
        ckpt = by_checkpoint.setdefault(row.checkpoint_id, {"datasets": [], "group_avgs": {}})
        ckpt["datasets"].append(row.__dict__)
    for checkpoint_id, payload in by_checkpoint.items():
        for group in sorted({row.group for row in rows if row.checkpoint_id == checkpoint_id}):
            values = [
                row.primary_value
                for row in rows
                if row.checkpoint_id == checkpoint_id
                and row.group == group
                and row.status == "ok"
                and row.primary_value is not None
            ]
            payload["group_avgs"][group] = sum(values) / len(values) if values else None

    simple: dict[str, dict[str, Any]] = {}
    for checkpoint_id, payload in sorted(by_checkpoint.items()):
        checkpoint_simple: dict[str, Any] = {"datasets": {}, "groups": {}}
        for row in rows:
            if row.checkpoint_id != checkpoint_id:
                continue
            checkpoint_simple["datasets"][row.dataset] = row.primary_value
        for group, value in sorted(payload["group_avgs"].items()):
            checkpoint_simple["groups"][group] = value
        simple[checkpoint_id] = checkpoint_simple

    _write_json(run_dir / "summary.json", {"checkpoints": by_checkpoint})
    _write_json(run_dir / "summary_simple.json", simple)

    simple_lines: list[str] = []
    for checkpoint_id, payload in simple.items():
        simple_lines.append(f"{checkpoint_id}:")
        for dataset, value in payload["datasets"].items():
            rendered = "failed" if value is None else f"{value:.6f}"
            simple_lines.append(f'  "{dataset}": "{rendered}"')
        for group, value in payload["groups"].items():
            rendered = "failed" if value is None else f"{value:.6f}"
            simple_lines.append(f'  "{group}": "{rendered}"')
        simple_lines.append("")
    (run_dir / "summary_simple.txt").write_text(
        "\n".join(simple_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    lines = ["# Offline Evaluation Summary", ""]
    lines.append("## Dataset Metrics")
    lines.append("")
    lines.append("| Checkpoint | Group | Dataset | Primary | Secondary | Status |")
    lines.append("|---|---|---|---:|---:|---|")
    for row in rows:
        primary = "" if row.primary_value is None else f"{row.primary_value:.6f}"
        secondary = "" if row.secondary_value is None else f"{row.secondary_value:.6f}"
        lines.append(
            f"| {row.checkpoint_id} | {row.group} | {row.dataset} | "
            f"{primary} | {secondary} | {row.status} |"
        )
    lines.append("")
    lines.append("## Group Averages")
    lines.append("")
    lines.append("| Checkpoint | Math | Natural Language | Coding |")
    lines.append("|---|---:|---:|---:|")
    for checkpoint_id, payload in sorted(by_checkpoint.items()):
        avgs = payload["group_avgs"]
        math_avg = "" if avgs.get("math") is None else f"{avgs['math']:.6f}"
        nl_avg = "" if avgs.get("nl") is None else f"{avgs['nl']:.6f}"
        coding_avg = "" if avgs.get("coding") is None else f"{avgs['coding']:.6f}"
        lines.append(f"| {checkpoint_id} | {math_avg} | {nl_avg} | {coding_avg} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_queue(queue_path: Path) -> None:
    raw = _read_yaml(queue_path)
    config = _resolve_config(queue_path, raw)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(queue_path, run_dir / "queue.input.yaml")
    _write_json(run_dir / "queue.resolved.json", config)
    (run_dir / "queue.resolved.yaml").write_text(
        _safe_dump_yaml(config),
        encoding="utf-8",
    )

    python_bin = _find_python([Path(config["python_bin"])])
    config["python_bin"] = str(python_bin)

    groups = set(config["groups"])
    subset_outputs = prepare_all(
        data_root=Path(config["data_root"]),
        subset_dir=SUBSET_DIR,
        seed=int(config["subset_seed"]),
        mmlu_size=int(config["mmlu_subset_size"]),
        amc23_size=config.get("math_row_limit") if "math" in groups else None,
        aime24_size=config.get("math_row_limit") if "math" in groups else None,
        math500_size=int(config["math500_subset_size"]),
        gsm8k_size=int(config["gsm8k_subset_size"]),
        gpqa_size=int(config["gpqa_subset_size"]),
        lcb_release=str(config["livecodebench_release"]),
        lcb_size=int(config["livecodebench_subset_size"]),
        lcb_local_dataset_path=Path(config["livecodebench_local_dataset_path"]),
        include_math500="math" in groups and config.get("math500_subset_size") not in (None, "", "null", "None"),
        include_gsm8k="math" in groups and config.get("gsm8k_subset_size") not in (None, "", "null", "None"),
        include_mmlu="nl" in groups,
        include_gpqa="nl" in groups and config.get("gpqa_subset_size") not in (None, "", "null", "None"),
        include_lcb="coding" in groups,
        force=bool(config["force_prepare_subsets"]),
    )
    _write_json(run_dir / "subsets.json", subset_outputs)

    checkpoints = raw.get("checkpoints") or []
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Queue config must contain a non-empty `checkpoints` list")

    all_rows: list[EvalResult] = []
    for item in checkpoints:
        checkpoint_id = _safe_id(str(item["id"]))
        checkpoint = Path(item["path"]).expanduser()
        checkpoint_kind = _checkpoint_kind(checkpoint)
        if checkpoint_kind == "unknown":
            error = f"Unsupported checkpoint format for {checkpoint}"
            all_rows.append(
                EvalResult(checkpoint_id, "setup", "checkpoint", "", None, None, None, "failed", "", "", error)
            )
            _write_summaries(run_dir, all_rows)
            if config["fail_fast"]:
                raise RuntimeError(error)
            continue

        try:
            if "math" in groups:
                data_root = Path(config["data_root"])
                math_row_limit = config.get("math_row_limit")
                math500_limit = _min_optional_limit(
                    config.get("math500_subset_size"),
                    math_row_limit,
                )
                gsm8k_limit = _min_optional_limit(
                    config.get("gsm8k_subset_size"),
                    math_row_limit,
                )
                val_files = [
                    _maybe_limited_fixed_subset(
                        data_root / "amc23_1010.parquet",
                        subset_outputs,
                        "amc23",
                        math_row_limit,
                    ),
                    _maybe_limited_fixed_subset(
                        data_root / "aime24_0726.parquet",
                        subset_outputs,
                        "aime24",
                        math_row_limit,
                    ),
                    _maybe_limited_fixed_subset(
                        data_root / "math500.parquet",
                        subset_outputs,
                        "math500",
                        math500_limit,
                    ),
                    _maybe_limited_fixed_subset(
                        Path(subset_outputs["gsm8k"]),
                        subset_outputs,
                        "gsm8k",
                        gsm8k_limit,
                    ),
                ]
                all_rows.extend(
                    _run_verl_group(
                        config,
                        checkpoint_id,
                        checkpoint,
                        "math",
                        val_files,
                        int(config["math_n"]),
                        run_dir,
                    )
                )
                _write_summaries(run_dir, all_rows)

            if "nl" in groups:
                nl_row_limit = config.get("nl_row_limit")
                gpqa_limit = _min_optional_limit(
                    config.get("gpqa_subset_size"),
                    nl_row_limit,
                )
                val_files = [
                    _maybe_limited_fixed_subset(
                        Path(subset_outputs["mmlu_pro"]),
                        subset_outputs,
                        "mmlu_pro",
                        nl_row_limit,
                    ),
                    _maybe_limited_fixed_subset(
                        Path(config["data_root"]) / "gpqa_diamond_eval.parquet",
                        subset_outputs,
                        "gpqa_diamond",
                        gpqa_limit,
                    ),
                ]
                all_rows.extend(
                    _run_verl_group(
                        config,
                        checkpoint_id,
                        checkpoint,
                        "nl",
                        val_files,
                        int(config["nl_n"]),
                        run_dir,
                    )
                )
                _write_summaries(run_dir, all_rows)

            if "coding" in groups:
                all_rows.append(
                    _run_livecodebench(
                        config,
                        checkpoint_id,
                        checkpoint,
                        Path(subset_outputs["livecodebench"]),
                        run_dir,
                    )
                )
                _write_summaries(run_dir, all_rows)
        except Exception as exc:
            all_rows.append(
                EvalResult(
                    checkpoint_id=checkpoint_id,
                    group="setup",
                    dataset="checkpoint",
                    primary_metric="",
                    primary_value=None,
                    secondary_metric=None,
                    secondary_value=None,
                    status="failed",
                    metric_file="",
                    log_file="",
                    error=str(exc),
                )
            )
            _write_summaries(run_dir, all_rows)
            if config["fail_fast"]:
                raise

    _write_summaries(run_dir, all_rows)
    print(f"[OfflineEval] Summary: {run_dir / 'summary.md'}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m evaluation.run_offline_eval_queue <queue.yaml>")
    run_queue(Path(sys.argv[1]).expanduser().resolve())


if __name__ == "__main__":
    main()
