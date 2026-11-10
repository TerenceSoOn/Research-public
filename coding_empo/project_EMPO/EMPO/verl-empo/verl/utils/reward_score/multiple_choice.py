"""Conservative multiple-choice answer scoring for EMPO validation.

This scorer is intentionally stricter than a whole-response letter search. It
first prefers boxed answers, then only inspects the final-answer region.
"""

import re
from typing import Any


VALID_CHOICES = set("ABCDEFGHIJ")
FINAL_REGION_CHARS = 800


def _strip_latex_text_wrappers(text: str) -> str:
    previous = None
    current = text
    wrapper_re = re.compile(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]+)\}")
    while previous != current:
        previous = current
        current = wrapper_re.sub(r"\1", current)
    return current


def normalize_choice(value: Any) -> str | None:
    """Normalize a value to one option letter A-J, or None if ambiguous."""
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[-1]

    text = _strip_latex_text_wrappers(str(value)).strip()
    text = text.strip(" \t\r\n$`")

    patterns = [
        r"^[\(\[\{]?\s*([A-Ja-j])\s*[\)\]\}]?\.?$",
        r"^(?:option|choice|answer)\s*[:：]?\s*[\(\[\{]?\s*([A-Ja-j])\s*[\)\]\}]?\.?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _iter_latex_macro_contents(text: str, macros: tuple[str, ...]):
    starts = []
    for macro in macros:
        for match in re.finditer(re.escape(macro) + r"\s*\{", text):
            starts.append((match.start(), match.end()))

    for macro_start, content_start in sorted(starts):
        depth = 1
        pos = content_start
        while pos < len(text) and depth > 0:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            yield macro_start, text[content_start : pos - 1]


def extract_choice_answer(solution_str: str) -> tuple[str | None, str]:
    """Extract a final MCQ option letter without scanning the full response."""
    if solution_str is None:
        return None, "multiple_choice_not_found"

    solution_str = str(solution_str)

    boxed_candidates = list(
        _iter_latex_macro_contents(solution_str, ("\\boxed", "\\fbox"))
    )
    for _, candidate in reversed(boxed_candidates):
        choice = normalize_choice(candidate)
        if choice is not None:
            return choice, "multiple_choice_boxed"

    final_region = solution_str[-FINAL_REGION_CHARS:]
    final_lines = [line.strip() for line in final_region.splitlines() if line.strip()]

    marker_patterns = [
        r"(?:final\s+answer|answer|correct\s+answer)\s*(?:is|:|=)?\s*(?:option|choice)?\s*[\(\[]?\s*([A-Ja-j])\s*[\)\].]?\s*$",
        r"(?:therefore|thus|so)[,\s]*(?:the\s+answer\s+is\s*)?(?:option|choice)?\s*[\(\[]?\s*([A-Ja-j])\s*[\)\].]?\s*$",
    ]
    for line in reversed(final_lines[-8:]):
        for pattern in marker_patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper(), "multiple_choice_final_marker"

    for line in reversed(final_lines[-4:]):
        choice = normalize_choice(line)
        if choice is not None:
            return choice, "multiple_choice_final_line"

    return None, "multiple_choice_not_found"


def compute_score(solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> dict:
    """Score an A-J multiple-choice response."""
    gt = normalize_choice(ground_truth)
    pred, scored_by = extract_choice_answer(solution_str)

    if gt is None:
        return {
            "score": 0.0,
            "point": 0.0,
            "acc": False,
            "extracted_gt": str(ground_truth),
            "extracted_pred": pred or "",
            "scored_by": "multiple_choice_invalid_gt",
        }

    acc = pred == gt
    score = 1.0 if acc else 0.0
    return {
        "score": score,
        "point": score,
        "acc": acc,
        "extracted_gt": gt,
        "extracted_pred": pred or "",
        "scored_by": scored_by,
    }
