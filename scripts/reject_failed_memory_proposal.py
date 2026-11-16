#!/usr/bin/env python3
"""Reject a recorded, token-overflowed proposal without refunding its call budget."""

import argparse
import fcntl
from pathlib import Path

from rsi_base.storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("round", type=int)
    args = parser.parse_args()
    with (args.run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = args.run / "evolution" / f"{args.round:03}"
        if (
            (args.run / "frozen.json").exists()
            or (directory / "proposal.json").exists()
            or (directory / "decision.json").exists()
        ):
            raise ValueError("Refusing to change completed or valid proposal state")
        result = read_json(directory / "meta_result.json")
        if not result.get("is_error") or "output token maximum" not in result.get("result", ""):
            raise ValueError("Only a recorded output-token overflow can be rejected here")
        write_json(
            directory / "invalid.json",
            {
                "error_type": "MetaOutputOverflow",
                "record": "meta_result.json",
                "call_budget_refunded": False,
                "repair": "Reject missing proposal; retain raw error, usage and all task results",
            },
        )
        print(f"Rejected overflowed round {args.round}; no calls refunded")


if __name__ == "__main__":
    main()
