#!/usr/bin/env python3
"""Run one bounded, project-isolated Claude Code proposal."""

import json
from pathlib import Path

from rsi_base.improve import ClaudeMeta
from rsi_base.types import load_config

root = Path(__file__).resolve().parents[1]
config = load_config(root / "configs/deepseek_smoke.json")
output = root / ".local/claude_probe"
output.mkdir(parents=True, exist_ok=True)
proposal, metadata = ClaudeMeta(config.meta, root).propose(
    config.agent,
    [{"question": "What is 2+2?", "response": "4", "score": 0, "status": "format_error"}],
    output,
)
print(
    json.dumps(
        {"ok": True, "proposal_valid": True, "reason": proposal.reason, **metadata},
        ensure_ascii=False,
        indent=2,
    )
)
