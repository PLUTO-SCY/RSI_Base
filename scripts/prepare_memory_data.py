#!/usr/bin/env python3
"""Download only pinned benchmark data blobs; no third-party framework code is used."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import httpx

from rsi_base.storage import write_json

SOURCES = {
    "train": ("1e41fed548ba5bb76220260eed0e0f6d93d30e6d", 200),
    "val": ("e85fc0660a74405af7301cd8c9e1dbe0e93f51b8", 50),
    "test": ("ecb94153e9eec31818f92352d5b84374390ec2c0", 212),
}
LABELS = sorted(
    "drug reaction|allergy|chicken pox|diabetes|psoriasis|hypertension|cervical spondylosis|"
    "bronchial asthma|varicose veins|malaria|dengue|arthritis|impetigo|fungal infection|"
    "common cold|gastroesophageal reflux disease|urinary tract infection|typhoid|pneumonia|"
    "peptic ulcer disease|jaundice|migraine".split("|")
)
RAW_REVISION = "f805ad3aa4ada290458c1e0deadbe132fbbdea4e"


def download_blob(split, blob, *, get=httpx.get):
    url = f"https://api.github.com/repos/stanford-iris-lab/meta-harness/git/blobs/{blob}"
    try:
        response = get(url, timeout=60)
        response.raise_for_status()
    except httpx.HTTPError:
        # Public API quotas must not require each new user to configure a GitHub token.
        raw_url = (
            "https://raw.githubusercontent.com/stanford-iris-lab/meta-harness/"
            f"{RAW_REVISION}/reference_examples/text_classification/data/"
            f"symptom_diagnosis/{split}.jsonl"
        )
        response = get(raw_url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        data = response.content
    else:
        data = base64.b64decode(response.json()["content"])
    if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != blob:
        raise ValueError("Git data blob fingerprint mismatch")
    return data


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data/memory-s2d")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a dataset; use the existing pinned manifest")
    assembled, seen, records, exclusions = {}, {}, {}, []
    for split, (blob, count) in SOURCES.items():
        url = f"https://api.github.com/repos/stanford-iris-lab/meta-harness/git/blobs/{blob}"
        data = download_blob(split, blob)
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if len(rows) != count:
            raise ValueError("Unexpected split size")
        converted = []
        for index, row in enumerate(rows):
            question, answer = row["question"], row["answer"].strip().casefold()
            normalized = " ".join(question.split()).casefold()
            if not normalized or answer not in LABELS:
                raise ValueError("Empty question or unknown label")
            if normalized in seen:
                previous = seen[normalized]
                if split != "test":
                    raise ValueError("Unexpected training/validation duplicate")
                exclusions.append(
                    {
                        "id": f"s2d-{split}-{index:03}",
                        "duplicate_of": previous,
                        "rule": "normalized exact query overlap; retain earlier training/validation row",
                    }
                )
                continue
            seen[normalized] = f"s2d-{split}-{index:03}"
            converted.append(
                {"id": f"s2d-{split}-{index:03}", "question": question, "answer": answer}
            )
        assembled[split] = converted
        records[split] = {
            "git_blob": blob,
            "source": url,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_count": count,
            "count": len(converted),
        }
    args.output.mkdir(parents=True)
    for split, rows in assembled.items():
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (args.output / f"{split}.jsonl").write_bytes(data)
        records[split].update(
            sha256=hashlib.sha256(data).hexdigest(), ids=[row["id"] for row in rows]
        )
    write_json(args.output / "labels.json", LABELS)
    write_json(
        args.output / "manifest.json",
        {
            "benchmark": "Symptom2Disease: Meta-Harness/MCE released splits",
            "reference": "https://github.com/stanford-iris-lab/meta-harness/tree/main/reference_examples/text_classification/data/symptom_diagnosis",
            "splits": records,
            "exclusions": exclusions,
            "labels": LABELS,
            "scope": "Data-only reuse; independent runner; fixed small-model adaptation, not paper main-table reproduction",
        },
    )
    print(
        json.dumps(
            {
                "dataset": str(args.output),
                "counts": {k: len(v) for k, v in assembled.items()},
                "classes": len(LABELS),
            }
        )
    )


if __name__ == "__main__":
    main()
