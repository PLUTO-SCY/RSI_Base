#!/usr/bin/env python3
"""Compare native and served instruction handling on unrelated sanity prompts."""

import argparse
import json
from pathlib import Path

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=".local/models/qwen2.5-1.5b-instruct")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--output", type=Path, default=Path(".local/native-task-probe.json"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.directory)
    model = AutoModelForCausalLM.from_pretrained(args.directory, dtype=torch.bfloat16).to("cuda")
    client = httpx.Client(base_url=args.url + "/", trust_env=False, timeout=120)
    cases = [
        [
            {"role": "system", "content": "Reply with exactly PINEAPPLE and nothing else."},
            {"role": "user", "content": "What is 1+1?"},
        ],
        [{"role": "user", "content": "Reply with exactly PINEAPPLE and nothing else."}],
        [
            {
                "role": "system",
                "content": "Your final line must begin with #### and give the numeric answer.",
            },
            {"role": "user", "content": "What is 1+1?"},
        ],
    ]
    results = []
    with torch.inference_mode():
        for messages in cases:
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            ).to("cuda")
            response = model.generate(
                ids, max_new_tokens=80, do_sample=False, repetition_penalty=1.0
            )
            native = tokenizer.decode(response[0, ids.shape[-1] :], skip_special_tokens=True)
            served = client.post(
                "chat/completions",
                json={
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": 80,
                    "temperature": 0,
                    "repetition_penalty": 1.0,
                },
            )
            served.raise_for_status()
            results.append(
                {
                    "messages": messages,
                    "native": native,
                    "served": served.json()["choices"][0]["message"]["content"],
                }
            )
            print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps({"model": args.model, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
