from __future__ import annotations

import argparse
import json
from pathlib import Path

from bugscout.eval.metrics import mrr_at_k, recall_at_k


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank_jsonl", type=str, required=True)
    ap.add_argument("--candidate_n", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    rows = load_jsonl(args.rerank_jsonl, limit=limit)

    mrr10_sum = 0.0
    r10_sum = 0.0
    n = 0

    for row in rows:
        ranked_paths = row["candidate_paths"][: args.candidate_n]   # already BM25-sorted
        relevant = set(row["relevant_paths"])

        if not relevant:
            continue

        mrr10_sum += mrr_at_k(ranked_paths, relevant, 10)
        r10_sum += recall_at_k(ranked_paths, relevant, 10)
        n += 1

    metrics = {
        "mrr@10": mrr10_sum / n if n else 0.0,
        "recall@10": r10_sum / n if n else 0.0,
        "n": float(n),
        "candidate_n": float(args.candidate_n),
    }

    print("Metrics:", metrics)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()