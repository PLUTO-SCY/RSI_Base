"""Numeric exact-match reward for verl's generated responses."""

from rsi_base.math_scoring import math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != "rsi_numeric":
        raise ValueError(f"Unexpected GRPO data source: {data_source}")
    result = math_score(solution_str, ground_truth)
    return {
        "score": result["score"],
        "status": result["status"],
        "problem_id": extra_info["id"],
    }
