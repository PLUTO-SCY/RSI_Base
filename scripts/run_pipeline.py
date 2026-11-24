#!/usr/bin/env python3
"""Run a joint numeric experiment and test its frozen selection."""

import argparse
import json
from pathlib import Path

from rsi_base.pipeline import Pipeline, PipelineConfig
from rsi_base.storage import read_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    run = phases.add_parser("run", help="Evolve on the training and development splits")
    run.add_argument("--config", type=Path, default=root / "configs/pipeline.json")
    run.add_argument("--data", type=Path, default=root / "data/gsm8k-grpo")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    test = phases.add_parser("test", help="Compare initial and selected states on held-out tasks")
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "run":
        config = PipelineConfig.model_validate_json(args.config.read_text())
        result = Pipeline(config, args.data, args.output, root).run(resume=args.resume)
        result = {
            "initial_development_accuracy": result["initial_development"]["accuracy"],
            "selected_development_accuracy": result["selected_development"]["accuracy"],
            "selected_round": result["selected_round"],
            "rounds": [
                {key: round_[key] for key in ("round", "parent_round", "strategy", "status")}
                for round_ in result["rounds"]
            ],
            "frozen": str(args.output / "frozen.json"),
        }
    else:
        run_path = args.output / "run.json"
        if not run_path.is_file() or not (args.output / "frozen.json").is_file():
            parser.error("Complete pipeline run before testing")
        manifest = read_json(run_path)
        config = PipelineConfig.model_validate(manifest["config"])
        result = Pipeline(config, manifest["data"], args.output, root).test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
