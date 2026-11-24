"""Apply the configured benchmark verifier to verl's generated responses."""

from rsi_base.components import NUMERIC_SCORER, checked_score, load_callable


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != "rsi_numeric":
        raise ValueError(f"Unexpected GRPO data source: {data_source}")
    scorer = load_callable(extra_info.get("scorer", NUMERIC_SCORER))
    result = checked_score(scorer, solution_str, ground_truth)
    return {
        "score": result["score"],
        "status": result["status"],
        "problem_id": extra_info["id"],
    }
