"""Load explicitly configured Python components from trusted local configuration."""

from functools import cache
from importlib import import_module
from inspect import getsourcefile
from pathlib import Path

from pydantic import Field

from .loop import file_hash
from .types import StrictModel

IMPORT_SPEC = r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*:[A-Za-z_][\w]*$"
NUMERIC_LOADER = "rsi_base.benchmark:load_problems"
NUMERIC_SCORER = "rsi_base.math_scoring:math_score"


class Factory(StrictModel):
    factory: str = Field(pattern=IMPORT_SPEC)
    options: dict = Field(default_factory=dict)


@cache
def load_callable(spec):
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("Expected a Python component in module:callable format")
    module, name = spec.split(":", 1)
    item = getattr(import_module(module), name)
    if not callable(item):
        raise ValueError(f"Configured component is not callable: {spec}")
    return item


def callable_hash(item):
    source = getsourcefile(item)
    if source is None or not Path(source).is_file():
        raise ValueError("Configured components need an inspectable Python source file")
    return file_hash(source)


def checked_score(scorer, response, answer):
    result = scorer(response, answer)
    if (
        not isinstance(result, dict)
        or result.get("score") not in (0, 1)
        or not isinstance(result.get("status"), str)
    ):
        raise ValueError("Benchmark scorer must return a binary score and a string status")
    return result
