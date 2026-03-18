from __future__ import annotations

import json
from pathlib import Path

def main():
    results_dir = Path("results")
    rows = []

    for p in sorted(results_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append((p.stem, m.get("mrr@10", 0.0), m.get("recall@10", 0.0), m.get("n", 0)))

    print("| Experiment | MRR@10 | Recall@10 | n |")
    print("|---|---:|---:|---:|")
    for name, mrr, r10, n in rows:
        print(f"| {name} | {mrr:.4f} | {r10:.4f} | {n} |")

if __name__ == "__main__":
    main()