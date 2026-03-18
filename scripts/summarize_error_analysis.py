from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CANONICAL = [
    "Vague / missing details",
    "Runtime behavior not obvious in code text",
    "Candidate generation missed correct file",
    "File too long / relevant region truncated",
]


ALIASES = {
    "vague / missing details": "Vague / missing details",
    "vague/missing details": "Vague / missing details",
    "vague": "Vague / missing details",

    "runtime behavior not obvious in code text": "Runtime behavior not obvious in code text",
    "runtime": "Runtime behavior not obvious in code text",

    "candidate generation missed correct file": "Candidate generation missed correct file",
    "candidate-generation miss": "Candidate generation missed correct file",
    "candidate miss": "Candidate generation missed correct file",
    "candidate": "Candidate generation missed correct file",

    "file too long / relevant region truncated": "File too long / relevant region truncated",
    "truncation": "File too long / relevant region truncated",
    "truncated": "File too long / relevant region truncated",
}


def normalize_category(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    key = s.lower()
    return ALIASES.get(key, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/error_analysis.csv")
    args = ap.parse_args()

    path = Path(args.csv)
    counts = Counter()
    blank = 0
    total = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cat = normalize_category(row.get("category", ""))
            if not cat:
                blank += 1
                continue
            counts[cat] += 1

    print("=== Error Analysis Summary ===")
    print("CSV:", path)
    print("Total rows:", total)
    print("Unlabeled rows:", blank)
    print()

    for cat in CANONICAL:
        print(f"{cat}: {counts.get(cat, 0)}")


if __name__ == "__main__":
    main()