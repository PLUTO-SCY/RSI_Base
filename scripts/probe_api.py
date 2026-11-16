#!/usr/bin/env python3
"""Explicit, small live checks; no credentials or full error bodies are printed."""

import json
from pathlib import Path

import httpx
from dotenv import dotenv_values
from openai import OpenAI

root = Path(__file__).resolve().parents[1]
values = dotenv_values(root / ".env.local")
key = values.get("RSI_API_KEY")
base = values.get("RSI_API_BASE_URL", "").rstrip("/")
if not key or not base:
    raise SystemExit("Configure .env.local first.")
model = "deepseek-v4.1-flash"
report = {"base_url": base, "model": model, "credential_configured": True}
try:
    client = OpenAI(base_url=base, api_key=key, timeout=60, max_retries=0)
    result = client.chat.completions.create(
        model=model,
        max_tokens=128,
        messages=[{"role": "user", "content": "Reply with exactly RSI_OK."}],
    )
    msg = result.choices[0].message
    report["chat_completions"] = {
        "ok": bool(msg.content),
        "response_model": result.model,
        "content": (msg.content or "")[:200],
        "finish_reason": result.choices[0].finish_reason,
        "usage": result.usage.model_dump() if result.usage else None,
    }
except Exception as exc:
    report["chat_completions"] = {
        "ok": False,
        "error_type": type(exc).__name__,
        "status": getattr(exc, "status_code", None),
    }
try:
    result = httpx.post(
        base + "/messages",
        headers={
            "x-api-key": key,
            "Authorization": "Bearer " + key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Reply with exactly RSI_OK."}],
        },
        timeout=60,
    )
    body = result.json()
    report["anthropic_messages"] = {
        "response_model": body.get("model"),
        "ok": result.is_success and body.get("type") == "message",
        "status": result.status_code,
        "type": body.get("type"),
        "error_type": body.get("error", {}).get("type")
        if isinstance(body.get("error"), dict)
        else None,
        "content": [x for x in body.get("content", []) if x.get("type") == "text"]
        if result.is_success
        else None,
        "usage": body.get("usage") if result.is_success else None,
    }
except Exception as exc:
    report["anthropic_messages"] = {"ok": False, "error_type": type(exc).__name__}
encoded = json.dumps(report, ensure_ascii=False, indent=2).replace(key, "[REDACTED]")
(root / ".local").mkdir(exist_ok=True)
(root / ".local/provider_probe.json").write_text(encoded + "\n", encoding="utf-8")
print(encoded)
raise SystemExit(
    0
    if all(report.get(name, {}).get("ok") for name in ("chat_completions", "anthropic_messages"))
    else 1
)
