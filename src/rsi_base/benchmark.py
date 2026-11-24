"""GSM8K-style numeric evaluation, separate from task inputs."""

import json
import re
from fractions import Fraction
from pathlib import Path


def load_problems(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    seen_ids, seen_questions = set(), set()
    for row in rows:
        if set(row) != {"id", "question", "answer"} or not all(
            isinstance(row[k], str) and row[k].strip() for k in row
        ):
            raise ValueError("Each row must have nonempty string id, question and answer")
        question = " ".join(row["question"].split()).casefold()
        if row["id"] in seen_ids or question in seen_questions:
            raise ValueError("Duplicate task ID or question")
        parse_number(row["answer"])
        seen_ids.add(row["id"])
        seen_questions.add(question)
    return rows


def parse_number(value):
    value = value.strip().replace(",", "").removeprefix("$").rstrip(".")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+|\d+/\d+)", value):
        raise ValueError("Not a numeric answer")
    return Fraction(value)


def score(response, expected):
    text = response.strip()
    if "####" in text:
        answer = text.rsplit("####", 1)[1].strip()
    elif re.fullmatch(r"\\boxed\{[^{}]+\}", text):
        answer = text[7:-1]
    else:
        answer = text
    try:
        parsed = parse_number(answer)
    except (ValueError, ZeroDivisionError):
        return {"score": 0, "status": "format_error", "parsed_answer": None}
    correct = parsed == parse_number(expected)
    return {
        "score": int(correct),
        "status": "correct" if correct else "wrong",
        "parsed_answer": str(parsed),
    }


def check_disjoint(dev, test):
    if {x["id"] for x in dev} & {x["id"] for x in test}:
        raise ValueError("Development and final task IDs overlap")

    def normalize(x):
        return " ".join(x["question"].split()).casefold()

    if {normalize(x) for x in dev} & {normalize(x) for x in test}:
        raise ValueError("Development and final questions overlap")
