import pytest

from rsi_base.math_scoring import math_score


@pytest.mark.parametrize(
    ("response", "answer"),
    [
        (r"Work here. \boxed{\frac{2}{4}}", "1/2"),
        (r"\boxed{-\dfrac{3}{6}}", "-1/2"),
        (r"Earlier \boxed{3}, corrected \boxed{4}", "4"),
        ("Work\n#### 0.25", "1/4"),
        ("0.25", "1/4"),
        (r"\boxed{8}" + "\n#### <number> 8", "8"),
    ],
)
def test_explicit_math_answers(response, answer):
    assert math_score(response, answer)["score"] == 1


@pytest.mark.parametrize(
    "response",
    [
        "The calculation uses 2 and 3, so perhaps something else.",
        r"\boxed{\frac{1}{0}}",
        r"\boxed{1+1}",
        r"\boxed{unfinished",
        r"\boxed{__import__('os').system('false')}",
    ],
)
def test_math_parser_rejects_implicit_symbolic_or_invalid_answers(response):
    assert math_score(response, "2")["status"] == "format_error"


def test_conflicting_final_formats_cannot_pick_a_favorable_answer():
    result = math_score(r"\boxed{2}" + "\n#### 3", "2")
    assert result["score"] == 0
    assert result["status"] == "conflicting_answers"


def test_math_score_retains_separate_strict_format_diagnostic():
    result = math_score(r"A derivation. \boxed{2}", "2")
    assert result["score"] == 1
    assert result["strict_format_compliant"] is False
