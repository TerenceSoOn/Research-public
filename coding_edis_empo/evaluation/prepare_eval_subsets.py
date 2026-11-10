"""Prepare fixed offline-evaluation subsets.

The math datasets are evaluated in full. This module materializes only the
large or runtime-capped subsets:
- stratified MMLU-Pro parquet subset
- fixed AMC23 parquet subset when a row cap is requested
- fixed AIME24 parquet subset when a row cap is requested
- fixed Math500 parquet subset
- fixed GSM8K parquet subset
- fixed GPQA-Diamond parquet subset
- fixed LiveCodeBench question-id subset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any


SEED = 20260418
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(os.environ.get("RUNTIME_ROOT", PROJECT_ROOT / "runtime")).expanduser()
SUBSET_DIR = PROJECT_ROOT / "evaluation" / "subsets"
MAX_MMLU_PRO_SUBSET_SIZE = 300
MAX_MATH500_SUBSET_SIZE = 300
MAX_GSM8K_SUBSET_SIZE = 300
MAX_GPQA_DIAMOND_SUBSET_SIZE = 300


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATA_ROOT = Path(os.environ["DATA_ROOT"]).expanduser() if "DATA_ROOT" in os.environ else _first_existing_path(
    [
        RUNTIME_ROOT / "math_data",
        Path("/home/pterence6699/coding_empo/project_EMPO/EMPO/math_data"),
    ]
)
DEFAULT_LCB_LOCAL_DATASET = (
    Path(os.environ["LIVECODEBENCH_LOCAL_DATASET_PATH"]).expanduser()
    if "LIVECODEBENCH_LOCAL_DATASET_PATH" in os.environ
    else _first_existing_path(
        [
            Path(os.environ.get("LCB_DATA_ROOT", RUNTIME_ROOT / "livecodebench")).expanduser()
            / "livecodebench_code_generation_lite_release_v6_test.parquet",
            SUBSET_DIR / "livecodebench_code_generation_lite_release_v6_test.parquet",
        ]
    )
)


def _json_default(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _extract_category(extra_info: Any) -> str:
    if isinstance(extra_info, dict):
        return str(extra_info.get("category") or extra_info.get("source") or "unknown")
    return "unknown"


def _allocate_stratified_counts(counts: dict[str, int], target_n: int) -> dict[str, int]:
    total = sum(counts.values())
    if target_n >= total:
        return dict(counts)

    raw = {key: target_n * value / total for key, value in counts.items()}
    allocated = {
        key: min(counts[key], max(1, int(raw[key])))
        for key in sorted(counts)
    }

    while sum(allocated.values()) > target_n:
        candidates = [key for key, value in allocated.items() if value > 1]
        key = min(candidates, key=lambda item: raw[item] - int(raw[item]))
        allocated[key] -= 1

    while sum(allocated.values()) < target_n:
        candidates = [key for key in counts if allocated[key] < counts[key]]
        key = max(candidates, key=lambda item: raw[item] - allocated[item])
        allocated[key] += 1

    return allocated


def prepare_mmlu_pro_subset(
    data_root: Path,
    subset_dir: Path,
    seed: int = SEED,
    sample_size: int = MAX_MMLU_PRO_SUBSET_SIZE,
    force: bool = False,
) -> Path:
    sample_size = min(int(sample_size), MAX_MMLU_PRO_SUBSET_SIZE)
    src = data_root / "mmlu_pro_eval.parquet"
    dst = subset_dir / f"mmlu_pro_eval_seed{seed}_n{sample_size}.parquet"
    meta_path = dst.with_suffix(".metadata.json")

    if dst.exists() and meta_path.exists() and not force:
        return dst
    if not src.exists():
        raise FileNotFoundError(f"MMLU-Pro eval parquet not found: {src}")

    import pandas as pd

    subset_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    if sample_size >= len(df):
        selected = df.copy()
        allocations = {"full": len(selected)}
    else:
        categories = df["extra_info"].map(_extract_category)
        category_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, category in categories.items():
            category_to_indices[category].append(idx)
        counts = {category: len(indices) for category, indices in category_to_indices.items()}
        allocations = _allocate_stratified_counts(counts, sample_size)

        selected_indices: list[int] = []
        for category, n_category in sorted(allocations.items()):
            category_seed = seed + zlib.crc32(category.encode("utf-8"))
            rng = random.Random(category_seed)
            indices = list(category_to_indices[category])
            rng.shuffle(indices)
            selected_indices.extend(indices[:n_category])
        selected_indices = sorted(selected_indices)
        selected = df.loc[selected_indices].copy()

    selected.to_parquet(dst, index=False)
    metadata = {
        "dataset": "mmlu_pro_eval",
        "source": str(src),
        "output": str(dst),
        "seed": seed,
        "sample_size": int(len(selected)),
        "requested_sample_size": int(sample_size),
        "stratified_by": "extra_info.category",
        "allocations": allocations,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return dst


def prepare_parquet_subset(
    src: Path,
    subset_dir: Path,
    dataset_name: str,
    seed: int = SEED,
    sample_size: int = 100,
    max_sample_size: int = 300,
    force: bool = False,
) -> Path:
    sample_size = min(int(sample_size), int(max_sample_size))
    dst = subset_dir / f"{dataset_name}_seed{seed}_n{sample_size}.parquet"
    meta_path = dst.with_suffix(".metadata.json")

    if dst.exists() and meta_path.exists() and not force:
        return dst
    if not src.exists():
        raise FileNotFoundError(f"{dataset_name} source parquet not found: {src}")

    import pandas as pd

    subset_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    if sample_size >= len(df):
        selected = df.copy()
        selected_indices = list(range(len(selected)))
    else:
        selected = df.sample(n=sample_size, random_state=seed).sort_index().copy()
        selected_indices = [int(item) for item in selected.index.tolist()]

    selected.to_parquet(dst, index=False)
    metadata = {
        "dataset": dataset_name,
        "source": str(src),
        "output": str(dst),
        "seed": seed,
        "sample_size": int(len(selected)),
        "requested_sample_size": int(sample_size),
        "selected_source_indices": selected_indices,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return dst


def _extract_gsm8k_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError(f"GSM8K answer does not contain {marker!r}: {answer[:120]}")
    return answer.split(marker, 1)[1].strip().replace(",", "")


def _gsm8k_eval_rows(examples: Any, seed: int = SEED) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    for idx, example in enumerate(examples):
        question = str(example["question"]).strip()
        answer = str(example["answer"]).strip()
        rows.append(
            {
                "data_source": "gsm8k",
                "prompt": [{"role": "user", "content": f"{question} {instruction}"}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": [_extract_gsm8k_answer(answer)],
                },
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "name": "gsm8k",
                    "question": question,
                    "answer": answer,
                    "seed": seed,
                },
            }
        )
    return rows


def _read_local_gsm8k_raw(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported GSM8K local raw format: {path}")
    required = {"question", "answer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"GSM8K raw file {path} is missing columns: {sorted(missing)}")
    return df[["question", "answer"]].to_dict("records")


def ensure_gsm8k_eval_dataset(
    data_root: Path,
    seed: int = SEED,
    force: bool = False,
) -> Path:
    dst = data_root / "gsm8k.parquet"
    if dst.exists() and not force:
        return dst

    raw_candidates = [
        data_root / "gsm8k_raw.parquet",
        data_root / "gsm8k_test.parquet",
        data_root / "gsm8k_raw.jsonl",
        data_root / "gsm8k_test.jsonl",
        data_root / "gsm8k_raw.json",
        data_root / "gsm8k_test.json",
    ]
    rows: list[dict[str, Any]] | None = None
    for raw_path in raw_candidates:
        if raw_path.exists():
            rows = _gsm8k_eval_rows(_read_local_gsm8k_raw(raw_path), seed=seed)
            break

    if rows is None:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "The `datasets` package is required to download GSM8K automatically. "
                "Alternatively place raw GSM8K as gsm8k_test.parquet/json/jsonl or "
                "eval-ready gsm8k.parquet under runtime/math_data."
            ) from exc

        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
        except Exception as exc:
            raise RuntimeError(
                "Could not download openai/gsm8k from Hugging Face. Place raw GSM8K "
                "as gsm8k_test.parquet/json/jsonl, or eval-ready gsm8k.parquet, under "
                f"{data_root}."
            ) from exc
        rows = _gsm8k_eval_rows(dataset, seed=seed)

    import pandas as pd

    data_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    return dst


def prepare_gsm8k_subset(
    data_root: Path,
    subset_dir: Path,
    seed: int = SEED,
    sample_size: int = 100,
    force: bool = False,
) -> Path:
    src = ensure_gsm8k_eval_dataset(data_root=data_root, seed=seed, force=False)
    return prepare_parquet_subset(
        src=src,
        subset_dir=subset_dir,
        dataset_name="gsm8k",
        seed=seed,
        sample_size=sample_size,
        max_sample_size=MAX_GSM8K_SUBSET_SIZE,
        force=force,
    )


def prepare_math500_subset(
    data_root: Path,
    subset_dir: Path,
    seed: int = SEED,
    sample_size: int = 100,
    force: bool = False,
) -> Path:
    return prepare_parquet_subset(
        src=data_root / "math500.parquet",
        subset_dir=subset_dir,
        dataset_name="math500",
        seed=seed,
        sample_size=sample_size,
        max_sample_size=MAX_MATH500_SUBSET_SIZE,
        force=force,
    )


def prepare_amc23_subset(
    data_root: Path,
    subset_dir: Path,
    seed: int = SEED,
    sample_size: int = 100,
    force: bool = False,
) -> Path:
    return prepare_parquet_subset(
        src=data_root / "amc23_1010.parquet",
        subset_dir=subset_dir,
        dataset_name="amc23_1010",
        seed=seed,
        sample_size=sample_size,
        max_sample_size=MAX_MATH500_SUBSET_SIZE,
        force=force,
    )


def prepare_aime24_subset(
    data_root: Path,
    subset_dir: Path,
    seed: int = SEED,
    sample_size: int = 100,
    force: bool = False,
) -> Path:
    return prepare_parquet_subset(
        src=data_root / "aime24_0726.parquet",
        subset_dir=subset_dir,
        dataset_name="aime24_0726",
        seed=seed,
        sample_size=sample_size,
        max_sample_size=MAX_MATH500_SUBSET_SIZE,
        force=force,
    )


def prepare_gpqa_diamond_subset(
    data_root: Path,
    subset_dir: Path,
    seed: int = SEED,
    sample_size: int = 100,
    force: bool = False,
) -> Path:
    return prepare_parquet_subset(
        src=data_root / "gpqa_diamond_eval.parquet",
        subset_dir=subset_dir,
        dataset_name="gpqa_diamond_eval",
        seed=seed,
        sample_size=sample_size,
        max_sample_size=MAX_GPQA_DIAMOND_SUBSET_SIZE,
        force=force,
    )


def prepare_livecodebench_subset(
    subset_dir: Path,
    release_version: str = "release_v6",
    seed: int = SEED,
    sample_size: int = 200,
    local_dataset_path: Path | None = None,
    force: bool = False,
) -> Path:
    dst = subset_dir / f"livecodebench_{release_version}_seed{seed}_n{sample_size}_ids.json"
    if dst.exists() and not force:
        return dst

    subset_dir.mkdir(parents=True, exist_ok=True)
    if local_dataset_path is not None and local_dataset_path.exists():
        import pandas as pd

        if local_dataset_path.suffix == ".parquet":
            df = pd.read_parquet(local_dataset_path, columns=["question_id"])
        elif local_dataset_path.suffix == ".jsonl":
            df = pd.read_json(local_dataset_path, lines=True)
        elif local_dataset_path.suffix == ".json":
            df = pd.read_json(local_dataset_path)
        else:
            raise ValueError(
                f"Unsupported LiveCodeBench local dataset format: {local_dataset_path}"
            )
        question_ids = sorted({str(item) for item in df["question_id"].tolist()})
        dataset_source = str(local_dataset_path)
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "The `datasets` package is required to prepare the LiveCodeBench subset."
            ) from exc

        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=release_version,
            split="test",
        )
        question_ids = sorted({str(row["question_id"]) for row in dataset})
        dataset_source = "livecodebench/code_generation_lite"

    if sample_size < len(question_ids):
        rng = random.Random(seed)
        selected_ids = sorted(rng.sample(question_ids, sample_size))
    else:
        selected_ids = question_ids

    payload = {
        "dataset": dataset_source,
        "release_version": release_version,
        "seed": seed,
        "sample_size": len(selected_ids),
        "requested_sample_size": sample_size,
        "question_ids": selected_ids,
    }
    dst.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return dst


def prepare_all(
    data_root: Path = DEFAULT_DATA_ROOT,
    subset_dir: Path = SUBSET_DIR,
    seed: int = SEED,
    mmlu_size: int = MAX_MMLU_PRO_SUBSET_SIZE,
    amc23_size: int | None = None,
    aime24_size: int | None = None,
    math500_size: int = 100,
    gsm8k_size: int = 100,
    gpqa_size: int = 100,
    lcb_release: str = "release_v6",
    lcb_size: int = 200,
    lcb_local_dataset_path: Path | None = None,
    include_mmlu: bool = True,
    include_math500: bool = True,
    include_gsm8k: bool = True,
    include_gpqa: bool = True,
    include_lcb: bool = True,
    force: bool = False,
) -> dict[str, str]:
    outputs = {}
    if amc23_size not in (None, "", "null", "None"):
        outputs["amc23"] = str(
            prepare_amc23_subset(
                data_root=data_root,
                subset_dir=subset_dir,
                seed=seed,
                sample_size=int(amc23_size),
                force=force,
            )
        )
    if aime24_size not in (None, "", "null", "None"):
        outputs["aime24"] = str(
            prepare_aime24_subset(
                data_root=data_root,
                subset_dir=subset_dir,
                seed=seed,
                sample_size=int(aime24_size),
                force=force,
            )
        )
    if include_math500:
        outputs["math500"] = str(
            prepare_math500_subset(
                data_root=data_root,
                subset_dir=subset_dir,
                seed=seed,
                sample_size=math500_size,
                force=force,
            )
        )
    if include_gsm8k:
        outputs["gsm8k"] = str(
            prepare_gsm8k_subset(
                data_root=data_root,
                subset_dir=subset_dir,
                seed=seed,
                sample_size=gsm8k_size,
                force=force,
            )
        )
    if include_mmlu:
        outputs["mmlu_pro"] = str(
            prepare_mmlu_pro_subset(
                data_root=data_root,
                subset_dir=subset_dir,
                seed=seed,
                sample_size=mmlu_size,
                force=force,
            )
        )
    if include_gpqa:
        outputs["gpqa_diamond"] = str(
            prepare_gpqa_diamond_subset(
                data_root=data_root,
                subset_dir=subset_dir,
                seed=seed,
                sample_size=gpqa_size,
                force=force,
            )
        )
    if include_lcb:
        outputs["livecodebench"] = str(
            prepare_livecodebench_subset(
                subset_dir=subset_dir,
                release_version=lcb_release,
                seed=seed,
                sample_size=lcb_size,
                local_dataset_path=lcb_local_dataset_path,
                force=force,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed offline eval subsets.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mmlu-size", type=int, default=MAX_MMLU_PRO_SUBSET_SIZE)
    parser.add_argument("--amc23-size", type=int, default=None)
    parser.add_argument("--aime24-size", type=int, default=None)
    parser.add_argument("--math500-size", type=int, default=100)
    parser.add_argument("--gsm8k-size", type=int, default=100)
    parser.add_argument("--gpqa-size", type=int, default=100)
    parser.add_argument("--lcb-release", default="release_v6")
    parser.add_argument("--lcb-size", type=int, default=200)
    parser.add_argument("--lcb-local-dataset-path", type=Path, default=DEFAULT_LCB_LOCAL_DATASET)
    parser.add_argument("--skip-lcb", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = prepare_all(
        data_root=args.data_root,
        subset_dir=args.subset_dir,
        seed=args.seed,
        mmlu_size=args.mmlu_size,
        amc23_size=args.amc23_size,
        aime24_size=args.aime24_size,
        math500_size=args.math500_size,
        gsm8k_size=args.gsm8k_size,
        gpqa_size=args.gpqa_size,
        lcb_release=args.lcb_release,
        lcb_size=args.lcb_size,
        lcb_local_dataset_path=args.lcb_local_dataset_path,
        include_mmlu=True,
        include_lcb=not args.skip_lcb,
        force=args.force,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
