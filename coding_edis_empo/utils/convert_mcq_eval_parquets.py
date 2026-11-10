#!/usr/bin/env python3
"""Convert raw MMLU-Pro and GPQA Diamond parquet files to Verl eval format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


CHOICES = "ABCDEFGHIJ"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DATA_ROOT = Path(
    os.environ.get("DATA_ROOT", PROJECT_ROOT / "runtime" / "math_data")
).expanduser()
LEGACY_RAW_DATA_ROOT = Path("/home/pterence6699/coding_edis/math_data")
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("DATA_ROOT", "/home/pterence6699/coding_empo/project_EMPO/EMPO/math_data")
).expanduser()
BOXED_INSTRUCTION = (
    "Think step by step. Put only the final option letter, from A to J, "
    "inside \\boxed{}."
)


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_choice(value) -> str:
    text = str(value).strip().upper()
    if len(text) == 1 and text in CHOICES:
        return text
    raise ValueError(f"Invalid answer choice: {value!r}")


def _prompt_message(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def _reward_model(answer: str) -> dict[str, str]:
    return {"style": "rule", "ground_truth": answer}


def _format_options(options: Iterable[str]) -> str:
    lines = []
    for idx, option in enumerate(options):
        if idx >= len(CHOICES):
            raise ValueError("MMLU-Pro row has more than 10 options; A-J is the supported range.")
        lines.append(f"{CHOICES[idx]}. {option}")
    return "\n".join(lines)


def convert_mmlu_pro(input_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(input_path)
    rows = []
    for idx, row in raw.iterrows():
        options = _as_list(row["options"])
        if not options:
            raise ValueError(f"MMLU-Pro row {idx} has no options.")

        answer = _normalize_choice(row["answer"])
        expected_from_index = CHOICES[int(row["answer_index"])]
        if answer != expected_from_index:
            raise ValueError(
                f"MMLU-Pro row {idx} answer mismatch: answer={answer}, "
                f"answer_index={row['answer_index']} -> {expected_from_index}"
            )

        prompt = (
            f"{row['question']}\n\n"
            f"Options:\n{_format_options(options)}\n\n"
            f"{BOXED_INSTRUCTION}"
        )
        rows.append(
            {
                "data_source": "mmlu_pro",
                "prompt": _prompt_message(prompt),
                "ability": "multiple_choice",
                "reward_model": _reward_model(answer),
                "extra_info": {
                    "index": int(idx),
                    "question_id": str(row.get("question_id", "")),
                    "question": str(row["question"]),
                    "options": options,
                    "answer_index": int(row["answer_index"]),
                    "category": str(row.get("category", "")),
                    "src": str(row.get("src", "")),
                },
            }
        )
    return pd.DataFrame(rows)


def convert_gpqa_diamond(input_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(input_path)
    rows = []
    for idx, row in raw.iterrows():
        answer = _normalize_choice(row["answer"])
        prompt = f"{row['question']}\n\n{BOXED_INSTRUCTION}"
        rows.append(
            {
                "data_source": "gpqa_diamond",
                "prompt": _prompt_message(prompt),
                "ability": "multiple_choice",
                "reward_model": _reward_model(answer),
                "extra_info": {
                    "index": int(idx),
                    "question": str(row["question"]),
                    "source": "gpqa_diamond",
                },
            }
        )
    return pd.DataFrame(rows)


def _write_and_report(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmlu-input",
        type=Path,
        default=_first_existing_path(
            [RUNTIME_DATA_ROOT / "mmlu_pro.parquet", LEGACY_RAW_DATA_ROOT / "mmlu_pro.parquet"]
        ),
    )
    parser.add_argument(
        "--gpqa-input",
        type=Path,
        default=_first_existing_path(
            [RUNTIME_DATA_ROOT / "gpqa_diamond.parquet", LEGACY_RAW_DATA_ROOT / "gpqa_diamond.parquet"]
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    mmlu_df = convert_mmlu_pro(args.mmlu_input)
    gpqa_df = convert_gpqa_diamond(args.gpqa_input)

    _write_and_report(mmlu_df, args.output_dir / "mmlu_pro_eval.parquet")
    _write_and_report(gpqa_df, args.output_dir / "gpqa_diamond_eval.parquet")


if __name__ == "__main__":
    main()
