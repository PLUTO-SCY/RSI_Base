"""Development-only tools: inspect real artifacts, probe code/policies and submit candidates."""

import json
import secrets
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import Field

from .classification import MemoryPolicy, effective_policy, normalize
from .memory_audit import paired_records
from .memory_loop import evaluate_policy, load_rows
from .storage import append_json, read_json, write_json
from .types import StrictModel, digest


class AgentSelection(StrictModel):
    candidate_ids: list[str] = Field(default_factory=list, max_length=2)


def tool_schema():
    definitions = [
        (
            "list_artifacts",
            "List available complete source, training memory and development artifacts. Final test and credentials are absent.",
            {},
        ),
        (
            "read_artifact",
            "Read a registered artifact with paging; no arbitrary filesystem access. Use offsets to read the entire file.",
            {
                "artifact": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 24000},
            },
        ),
        (
            "get_case",
            "Read complete model inputs/outputs, memory IDs and gold for a development case, including failed candidates.",
            {
                "candidate_id": {"type": "string"},
                "case_id": {"type": "string"},
                "repeat": {"type": "integer", "minimum": 0, "maximum": 2},
            },
        ),
        (
            "evaluate_candidate",
            "Execute a proposed policy/code on the fixed optimization probe batch; receive paired gains/losses and complete traces. Does not access final test.",
            {
                "policy": MemoryPolicy.model_json_schema(),
                "reason": {"type": "string", "maxLength": 1800},
            },
        ),
        (
            "submit_candidates",
            "Submit up to two already-probed nonduplicate candidate IDs for full development validation. Call before finishing.",
            {"candidate_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
        ),
    ]
    required = {
        "read_artifact": ["artifact"],
        "get_case": ["candidate_id", "case_id"],
        "evaluate_candidate": ["policy", "reason"],
        "submit_candidates": ["candidate_ids"],
    }
    return [
        {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required.get(name, []),
                "additionalProperties": False,
            },
        }
        for name, description, properties in definitions
    ]


class DevelopmentTools:
    def __init__(
        self,
        config,
        root,
        output,
        number,
        current,
        parent_records,
        training,
        development,
        probes,
        harness,
        task,
        ledger,
    ):
        self.config, self.root, self.output, self.number = config, Path(root), Path(output), number
        self.current, self.parent_records, self.harness, self.task, self.ledger = (
            current,
            parent_records,
            harness,
            task,
            ledger,
        )
        self.training, self.development, self.probes = training, development, probes
        self.references = {row["id"]: row for row in development}
        self.round_dir = self.output / "evolution" / f"{number:03}"
        self.round_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = {
            "source/" + p.name: p.read_text()
            for p in sorted((self.root / "src/rsi_base").glob("*.py"))
        }
        self.artifacts.update(
            {
                "memory/train.json": json.dumps(training, ensure_ascii=False),
                "development/cases.json": json.dumps(development, ensure_ascii=False),
                "parent/policy.json": current.model_dump_json(),
                "parent/records.jsonl": "\n".join(
                    json.dumps(r, ensure_ascii=False) for r in parent_records
                ),
            }
        )
        if config.model_context:
            self.artifacts["model/context.json"] = json.dumps(
                config.model_context, ensure_ascii=False
            )
        self.registry_path = self.output / "agent_registry.json"
        self.registry = read_json(self.registry_path) if self.registry_path.exists() else {}
        self.new = {
            key: value for key, value in self.registry.items() if key.startswith(f"r{number}-")
        }
        self.calls, self.attempts = 0, 0
        state_path = self.round_dir / "tool_budget.json"
        if state_path.exists():
            state = read_json(state_path)
            self.calls, self.attempts = state["calls"], state["attempts"]
        self.submitted = None
        self.probe_parent, self.probe_summary = evaluate_policy(
            self.round_dir / "parent_probe",
            probes,
            current,
            harness,
            task,
            ledger,
            repeats=1,
            seed=config.seed,
            concurrency=16,
        )
        self.parent_signatures = self.signatures(self.probe_parent)
        self.seen_signatures = {self.parent_signatures}
        self.full_signatures = set()
        if current.workflow == "direct":
            self.full_signatures.add(self.direct_signature(current))
        for item in self.new.values():
            self.seen_signatures.add(
                self.signatures(load_rows(self.output / item["probe"] / "results.jsonl"))
            )
            if item.get("full_request_signature"):
                self.full_signatures.add(item["full_request_signature"])
        for identifier, item in self.registry.items():
            self.artifacts[f"candidate/{identifier}/policy.json"] = json.dumps(item["policy"])
            self.artifacts[f"candidate/{identifier}/summary.json"] = json.dumps(item["summary"])
        calibration = self.output / "calibration.json"
        if calibration.exists():
            self.artifacts["controls/calibration.json"] = calibration.read_text()

    @staticmethod
    def signatures(records):
        return digest(
            sorted(
                (row["problem_id"], [trace.get("messages") for trace in row["traces"]])
                for row in records
            )
        )

    def direct_signature(self, policy):
        if policy.retrieval == "python":
            self.harness.index.prepare_code(
                policy, [(r["question"], None) for r in self.development]
            )
        suffix = getattr(getattr(self.task, "settings", None), "user_suffix", "")
        signatures = []
        for row in self.development:
            prompt, _ = self.harness.context(row["question"], policy)
            if suffix:
                prompt += "\n\n" + suffix
            signatures.append((row["id"], policy.system_prompt, prompt))
        return digest(signatures)

    def validate(self, policy):
        text = normalize(policy.system_prompt + "\n" + policy.retrieval_code)
        if any(
            normalize(row["question"]) in text or row["id"] in text
            for row in self.training + self.development
        ):
            raise ValueError("Case-specific queries/IDs cannot be embedded in a reusable candidate")
        if policy.retrieval == "python" and not policy.retrieval_code.strip():
            raise ValueError("Python retrieval requires def rank(query, memory)")

    def evaluate(self, policy, reason):
        self.attempts += 1
        write_json(
            self.round_dir / "tool_budget.json", {"calls": self.calls, "attempts": self.attempts}
        )
        if self.attempts > self.config.max_probes * 2 or len(self.new) >= self.config.max_probes:
            raise ValueError("Probe budget exhausted; submit already evaluated candidates")
        policy = effective_policy(MemoryPolicy.model_validate(policy))
        self.validate(policy)
        full_signature = self.direct_signature(policy) if policy.workflow == "direct" else None
        if full_signature is not None and full_signature in self.full_signatures:
            return {
                "duplicate_execution": True,
                "reason": "All full-development requests equal a previous policy; no Task calls made",
                "new_task_calls": 0,
            }
        identifier = f"r{self.number}-p{self.attempts}"
        location = self.round_dir / "probes" / identifier
        before = dict(self.ledger.used)
        records, summary = evaluate_policy(
            location,
            self.probes,
            policy,
            self.harness,
            self.task,
            self.ledger,
            repeats=1,
            seed=self.config.seed,
        )
        signature = self.signatures(records)
        if full_signature is not None:
            self.full_signatures.add(full_signature)
        self.seen_signatures.add(signature)
        pair = paired_records(self.probe_parent, records)
        item = {
            "policy": policy.model_dump(),
            "reason": reason,
            "probe": str(location.relative_to(self.output)),
            "summary": summary,
            "full_request_signature": full_signature,
        }
        self.new[identifier] = item
        self.registry[identifier] = item
        self.artifacts[f"candidate/{identifier}/policy.json"] = json.dumps(item["policy"])
        self.artifacts[f"candidate/{identifier}/summary.json"] = json.dumps(summary)
        write_json(location / "policy.json", policy.model_dump())
        write_json(self.registry_path, self.registry)
        losses = [key for key, value in pair["question_deltas"].items() if value < 0]
        gains = [key for key, value in pair["question_deltas"].items() if value > 0]
        return {
            "candidate_id": identifier,
            "summary": summary,
            "parent_summary": self.probe_summary,
            "gains": gains,
            "losses": losses,
            "new_task_calls": self.ledger.used["task"] - before["task"],
            "note": "Use get_case for any full parent/child trace. Probe scores do not select the final incumbent.",
        }

    def call(self, name, arguments):
        self.calls += 1
        write_json(
            self.round_dir / "tool_budget.json", {"calls": self.calls, "attempts": self.attempts}
        )
        if self.calls > self.config.max_tool_calls and name != "submit_candidates":
            raise ValueError("Tool budget exhausted; submit now")
        if name == "list_artifacts":
            return {
                "artifacts": [{"id": k, "characters": len(v)} for k, v in self.artifacts.items()],
                "candidate_ids": list(self.registry),
                "parent_failures": [
                    row["problem_id"]
                    for row in self.parent_records
                    if row["repeat"] == 0 and not row["score"]
                ],
                "probe_case_ids": [row["id"] for row in self.probes],
            }
        if name == "read_artifact":
            value = self.artifacts[arguments["artifact"]]
            offset, count = (
                max(0, int(arguments.get("offset", 0))),
                min(24000, max(100, int(arguments.get("max_chars", 16000)))),
            )
            return {
                "total_characters": len(value),
                "offset": offset,
                "text": value[offset : offset + count],
                "next_offset": offset + count if offset + count < len(value) else None,
            }
        if name == "get_case":
            identifier, case = arguments["candidate_id"], arguments["case_id"]
            repeat = int(arguments.get("repeat", 0))
            if identifier == "parent":
                records = self.parent_records
            else:
                item = self.registry[identifier]
                location = item.get("validation", item["probe"])
                records = load_rows(self.output / location / "results.jsonl")
            row = next(
                (r for r in records if r["problem_id"] == case and r["repeat"] == repeat), None
            )
            if row is None:
                raise ValueError(
                    "Case/repeat not present in this candidate's evaluated development batch"
                )
            return {"case": self.references[case], "trace": row}
        if name == "evaluate_candidate":
            return self.evaluate(arguments["policy"], arguments["reason"])
        if name == "submit_candidates":
            selected = AgentSelection.model_validate(arguments)
            if len(set(selected.candidate_ids)) != len(selected.candidate_ids) or any(
                identifier not in self.new for identifier in selected.candidate_ids
            ):
                raise ValueError("Submit only distinct IDs successfully probed in this round")
            self.submitted = selected.candidate_ids
            write_json(self.round_dir / "submitted.json", selected.model_dump())
            return {"accepted_submission": selected.candidate_ids}
        raise ValueError("Unknown host tool")


@contextmanager
def bridge(context):
    token = secrets.token_hex(32)
    mutex = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(403)
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if request["method"] == "tools/list":
                    response = {"tools": tool_schema()}
                else:
                    params = request["params"]
                    with mutex:
                        try:
                            result = context.call(params["name"], params.get("arguments", {}))
                            response = {
                                "content": [
                                    {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                                ]
                            }
                            append_json(
                                context.round_dir / "tool_calls.jsonl",
                                {
                                    "tool": params["name"],
                                    "arguments": params.get("arguments", {}),
                                    "result": result,
                                },
                            )
                        except Exception as exc:
                            response = {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": type(exc).__name__ + ": " + str(exc)[:1500],
                                    }
                                ],
                                "isError": True,
                            }
                            append_json(
                                context.round_dir / "tool_calls.jsonl",
                                {
                                    "tool": params["name"],
                                    "arguments": params.get("arguments", {}),
                                    "error": type(exc).__name__,
                                    "detail": str(exc)[:1500],
                                },
                            )
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (ValueError, KeyError):
                self.send_error(400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        port = server.server_address[1]
        config = {
            "mcpServers": {
                "rsi": {
                    "command": sys.executable,
                    "args": ["-m", "rsi_base.meta_tools", "--url", f"http://127.0.0.1:{port}/"],
                    "env": {"PYTHONPATH": str(context.root / "src")},
                }
            }
        }
        yield config, {"RSI_AGENT_TOKEN": token}, [tool["name"] for tool in tool_schema()]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
