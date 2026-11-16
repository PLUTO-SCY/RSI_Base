"""One model call per task. Reference answers never enter model requests."""

import os
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI


def credentials(root):
    values = {**dotenv_values(Path(root) / ".env.local")}
    for name in list(values) + ["RSI_API_KEY"]:
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


class TaskAgent:
    def __init__(self, settings, root):
        self.settings = settings
        key = os.environ.get(settings.api_key_env) or credentials(root).get(settings.api_key_env)
        if not key:
            raise ValueError(f"Missing credential: {settings.api_key_env}")
        self.client = OpenAI(
            base_url=settings.base_url, api_key=key, timeout=settings.timeout, max_retries=0
        )

    def run(self, spec, question):
        messages = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": question},
        ]
        started = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            extra_body=self.settings.extra_body,
        )
        return {
            "messages": messages,
            "response": response.choices[0].message.content or "",
            "response_model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
            "elapsed_seconds": time.monotonic() - started,
        }
