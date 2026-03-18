from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bugscout.models.dual_encoder import DualEncoder, DualEncoderConfig


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def rerank_topk(
    model,
    tok,
    row: dict,
    *,
    device: torch.device,
    candidate_n: int,
    max_len_issue: int,
    max_len_file: int,
    file_batch_size: int,
) -> list[str]:
    issue_text = row["issue_text"]
    cand_paths = row["candidate_paths"][:candidate_n]
    cand_texts = row["candidate_texts"][:candidate_n]

    issue_tok = tok([issue_text], padding=True, truncation=True, max_length=max_len_issue, return_tensors="pt")
    issue_tok = {k: v.to(device) for k, v in issue_tok.items()}
    issue_emb = model.encode_issue(issue_tok)[0]

    scores = []
    for i in range(0, len(cand_texts), file_batch_size):
        chunk = cand_texts[i : i + file_batch_size]
        file_tok = tok(chunk, padding=True, truncation=True, max_length=max_len_file, return_tensors="pt")
        file_tok = {k: v.to(device) for k, v in file_tok.items()}
        file_emb = model.encode_file(file_tok)
        scores.extend((file_emb @ issue_emb).detach().cpu().tolist())

    ranked_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
    return [cand_paths[j] for j in ranked_idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--rerank_jsonl", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="results/error_analysis.csv")

    ap.add_argument("--candidate_n", type=int, default=100)
    ap.add_argument("--max_len_issue", type=int, default=256)
    ap.add_argument("--max_len_file", type=int, default=256)
    ap.add_argument("--file_batch_size", type=int, default=32)

    ap.add_argument("--num_failures", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = DualEncoderConfig(**ckpt["cfg"])
    model = DualEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    tokenizer_name = ckpt.get("tokenizer_name", cfg.model_name)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    rows = load_jsonl(args.rerank_jsonl)

    failures = []
    for row in rows:
        relevant = set(row["relevant_paths"])
        ranked = rerank_topk(
            model, tok, row,
            device=device,
            candidate_n=args.candidate_n,
            max_len_issue=args.max_len_issue,
            max_len_file=args.max_len_file,
            file_batch_size=args.file_batch_size,
        )

        # Find first relevant rank
        first_rel = None
        for i, p in enumerate(ranked[:args.candidate_n], start=1):
            if p in relevant:
                first_rel = i
                break

        # Failure = not in top 10
        if first_rel is None or first_rel > 10:
            failures.append({
                "example_id": row.get("example_id", ""),
                "repo_id": row.get("repo_id", ""),
                "base_sha": row.get("base_sha", ""),
                "issue_text": row.get("issue_text", "")[:1000].replace("\n", " "),
                "relevant_paths": ";".join(row.get("relevant_paths", [])),
                "top10_pred": ";".join(ranked[:10]),
                "first_relevant_rank": first_rel if first_rel is not None else "",
                "category": "",        # you fill in
                "notes": "",           # you fill in
                "proposed_fix": "",    # you fill in
            })

    rng = random.Random(args.seed)
    rng.shuffle(failures)
    failures = failures[: args.num_failures]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()) if failures else [])
        writer.writeheader()
        for r in failures:
            writer.writerow(r)

    print(f"Exported {len(failures)} failures -> {out_path}")


if __name__ == "__main__":
    main()
