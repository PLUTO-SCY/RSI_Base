"""Command-line entry points. Run from the project root or pass --project-root."""

import argparse
import json
from pathlib import Path

from .loop import run
from .storage import redact
from .types import load_config


def main():
    parser = argparse.ArgumentParser(prog="rsi")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baseline", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--dev", type=Path, required=True)
        command.add_argument("--test", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--project-root", type=Path, default=Path.cwd())
        command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(
            load_config(args.config),
            args.dev,
            args.test,
            args.output,
            args.project_root,
            resume=args.resume,
            baseline=args.command == "baseline",
        )
    except Exception as exc:
        print(
            redact(
                json.dumps(
                    {"ok": False, "error_type": type(exc).__name__, "detail": str(exc)[:400]},
                    ensure_ascii=False,
                )
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
