#!/usr/bin/env python3
"""Cache pinned primary paper HTML for protocol review; never execute fetched content."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx

PAPERS = {
    "meta-harness": "2603.28052v1",
    "aflow": "2410.10762v3",
    "gepa": "2507.19457v2",
    "harnessforge": "2606.01779v1",
    "whale": "2609.00196v1",
    "dgm": "2505.22954v1",
    "adas": "2408.08435v2",
    "sica": "2504.15228v1",
    "rrsi": "2609.24972v1",
    "toolhop": "2501.02506v2",
}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1
        if tag in {"p", "div", "section", "tr", "h1", "h2", "h3", "h4", "li"}:
            self.parts.append("\n")
        if tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_data(self, value):
        if not self.skip:
            self.parts.append(value)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / ".local/research/harness-benchmarks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, version = item
        url = f"https://arxiv.org/html/{version}"
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        extractor = Text()
        extractor.feed(response.text)
        text = "\n".join(
            " ".join(line.split()) for line in "".join(extractor.parts).splitlines() if line.strip()
        )
        versions = re.findall(r"arXiv:\s*(\d+\.\d+v\d+)", text)
        if version not in versions:
            raise ValueError(f"Paper version not found: {name}")
        (args.out / f"{name}.html").write_bytes(response.content)
        (args.out / f"{name}.txt").write_text(text)
        return {
            "name": name,
            "version": version,
            "url": url,
            "html_sha256": hashlib.sha256(response.content).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, PAPERS.items()))
    manifest = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sources": records,
        "scope": "Download/fingerprint only; reading scope is documented separately; third-party code not executed",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"papers": len(records), "manifest": str(args.out / "manifest.json")}))


if __name__ == "__main__":
    main()
