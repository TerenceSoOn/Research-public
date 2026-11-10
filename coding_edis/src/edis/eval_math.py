#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math answer extraction and equivalence checking.
Aligned with EMPO eval logic: boxed{} extraction + numeric/symbolic equivalence.
"""

from __future__ import annotations

import re
import regex
from typing import Iterable, Optional
from math import isclose

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy


def find_box(pred_str: str) -> str:
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def _convert_pi_to_number(code_string: str) -> str:
    code_string = code_string.replace("\\pi", "π")
    code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
    code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
    code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
    code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
    return code_string


def clean_units(pred_str: str) -> str:
    pred_str = _convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        pred = find_box(pred_str)
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            pred = pred[-1] if len(pred) >= 1 else ""
        else:
            pred = ""

    return pred.strip()


def _parse_digits(num: str) -> Optional[float]:
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                return None
    return None


def _is_digit(num: str) -> bool:
    return _parse_digits(num) is not None


def _numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _symbolic_equal(a: str, b: str) -> bool:
    def _parse(s: str):
        for f in (parse_latex, parse_expr, latex2sympy):
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if _numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    try:
        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def math_equal(
    prediction: str,
    reference: str,
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    if prediction is None or reference is None:
        return False

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    if prediction.lower() == reference.lower():
        return True

    try:
        if _is_digit(prediction) and _is_digit(reference):
            prediction = _parse_digits(prediction)
            reference = _parse_digits(reference)
            if prediction is None or reference is None:
                return False
            gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
            for item in gt_result:
                if item is None:
                    continue
                if is_close:
                    if _numeric_equal(prediction, item):
                        return True
                else:
                    if item == prediction:
                        return True
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")

    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")

    if pred_str.lower() == ref_str.lower():
        return True

    if regex.match(r"(\(|\[).+(\)|\])", prediction) and regex.match(
        r"(\(|\[).+(\)|\])", reference
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))):
                return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if _symbolic_equal(pred, ref) or _symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True

    return _symbolic_equal(clean_units(prediction), clean_units(reference))


def is_correct(pred_str: str, ground_truths: Iterable[str]) -> bool:
    pred = extract_answer(pred_str)
    if pred is None:
        return False
    for gt in ground_truths:
        if math_equal(pred, gt):
            return True
    return False
