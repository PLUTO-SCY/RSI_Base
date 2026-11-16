"""Numeric subset of MATH: explicit final marker or balanced final boxed answer."""

import re

from .benchmark import parse_number, score


def parse_math_number(value):
    value = value.strip().strip("$").replace("−", "-")
    for token in [r"\left", r"\right", r"\,", r"\!"]:
        value = value.replace(token, "")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", r"(\1/\2)", value)
    # Only numeric signs and a single rational are admitted. Never eval model text.
    match = re.fullmatch(r"([+-]?)\(([+-]?\d+)/([+-]?\d+)\)", value)
    if match:
        from fractions import Fraction

        return (-1 if match[1] == "-" else 1) * Fraction(int(match[2]), int(match[3]))
    return parse_number(value)


def math_score(response, expected):
    strict = score(response, expected)
    marker, boxed = response.rfind("####"), response.rfind(r"\boxed{")
    values = {}
    if marker >= 0:
        try:
            values["marker"] = parse_math_number(response[marker + 4 :])
        except (ValueError, ZeroDivisionError):
            pass
    if boxed >= 0:
        start, depth, end = boxed + 7, 1, None
        for index in range(start, len(response)):
            depth += (response[index] == "{") - (response[index] == "}")
            if depth == 0:
                end = index
                break
        if end is not None:
            try:
                values["boxed"] = parse_math_number(response[start:end])
            except (ValueError, ZeroDivisionError):
                pass
    if marker < 0 and boxed < 0:
        try:
            values["number"] = parse_math_number(response)
        except (ValueError, ZeroDivisionError):
            pass
    metadata = {"strict_format_compliant": strict["status"] != "format_error"}
    if len(set(values.values())) > 1:
        return {
            "score": 0,
            "status": "conflicting_answers",
            "parsed_answer": None,
            "answer_format": "conflict",
            **metadata,
        }
    if not values:
        return {
            "score": 0,
            "status": "format_error",
            "parsed_answer": None,
            "answer_format": "unparsed",
            **metadata,
        }
    format_name = "boxed" if "boxed" in values else next(iter(values))
    parsed = values[format_name]
    correct = parsed == parse_number(expected)
    return {
        "score": int(correct),
        "status": "correct" if correct else "wrong",
        "parsed_answer": str(parsed),
        "answer_format": format_name,
        **metadata,
    }
