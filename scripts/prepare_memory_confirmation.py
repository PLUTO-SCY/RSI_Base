#!/usr/bin/env python3
"""Seal unused upstream examples before any test scores are read; never add them to memory."""

import hashlib
import json
from pathlib import Path

import httpx

from rsi_base.classification import normalize
from rsi_base.loop import file_hash
from rsi_base.memory_loop import load_rows
from rsi_base.storage import write_json

REVISION = "722cfb0e11f8ae37339c7f573b5e10429b94df49"


def trigrams(text):
    return {text[i : i + 3] for i in range(max(0, len(text) - 2))}


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/memory-confirmation"
    if output.exists():
        raise ValueError("Refusing to overwrite a sealed confirmation set")
    url = f"https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/resolve/{REVISION}/train.jsonl"
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    source = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    if len(source) != 853:
        raise ValueError("Unexpected upstream training pool size")
    base = root / "data/memory-s2d"
    labels = set(json.loads((base / "labels.json").read_text()))
    known = {
        normalize(row["question"])
        for split in ["train", "val", "test"]
        for row in load_rows(base / f"{split}.jsonl")
    }
    seen = set(known)
    known_grams = [(text, trigrams(text)) for text in known]
    rows, exclusions = [], []
    for index, row in enumerate(source):
        text, answer = row["input_text"], normalize(row["output_text"])
        normalized = normalize(text)
        reason = None
        if answer not in labels:
            reason = "outside fixed label set"
        elif normalized in seen:
            reason = "normalized exact overlap with an existing split or earlier source row"
        else:
            chars = trigrams(normalized)
            if any(
                len(chars & other) / max(1, len(chars | other)) >= 0.9 for _, other in known_grams
            ):
                reason = "character-trigram Jaccard >= 0.9 with a prior split/retained row"
        if reason:
            exclusions.append({"source_index": index, "reason": reason})
            continue
        rows.append({"id": f"s2d-confirm-{index:04}", "question": text, "answer": answer})
        seen.add(normalized)
        known_grams.append((normalized, chars))
    output.mkdir(parents=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (output / "test.jsonl").write_bytes(data)
    write_json(
        output / "manifest.json",
        {
            "source": url,
            "revision": REVISION,
            "source_sha256": hashlib.sha256(response.content).hexdigest(),
            "source_count": len(source),
            "retained_count": len(rows),
            "prior_data_manifest_sha256": file_hash(base / "manifest.json"),
            "test_sha256": hashlib.sha256(data).hexdigest(),
            "ids": [row["id"] for row in rows],
            "exclusions": exclusions,
            "source_role": "Unused upstream training rows reassigned to confirmation only; never enter training memory, Meta feedback or selection",
            "scope": "Additional in-distribution synthetic public data; separate from the released 211-question test; not OOD or a standard benchmark score",
            "selection_rule": "All unused allowed-label rows in source order after fixed exact and >=0.9 character-trigram deduplication; no model outcomes consulted",
        },
    )
    print(
        json.dumps(
            {
                "upstream": len(source),
                "confirmation": len(rows),
                "excluded": len(exclusions),
                "test_scores_consulted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
