from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bugscout.models.dual_encoder import DualEncoder, DualEncoderConfig
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


@torch.no_grad()
def eval_checkpoint(
    ckpt_path: str,
    rerank_jsonl: str,
    *,
    candidate_n: int,
    max_len_issue: int,
    max_len_file: int,
    file_batch_size: int,
    limit: int | None,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = DualEncoderConfig(**ckpt["cfg"])
    model = DualEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    tokenizer_name = ckpt.get("tokenizer_name", cfg.model_name)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    rows = load_jsonl(rerank_jsonl, limit=limit)

    mrr10_sum = 0.0
    r10_sum = 0.0
    n = 0

    for row in rows:
        issue_text = row["issue_text"]
        cand_paths = row["candidate_paths"][:candidate_n]
        cand_texts = row["candidate_texts"][:candidate_n]
        relevant = set(row["relevant_paths"])

        # Encode issue
        issue_tok = tok([issue_text], padding=True, truncation=True, max_length=max_len_issue, return_tensors="pt")
        issue_tok = {k: v.to(device) for k, v in issue_tok.items()}
        issue_emb = model.encode_issue(issue_tok)[0]  # [D]

        # Encode candidates in batches
        scores = []
        for i in range(0, len(cand_texts), file_batch_size):
            chunk = cand_texts[i : i + file_batch_size]
            file_tok = tok(chunk, padding=True, truncation=True, max_length=max_len_file, return_tensors="pt")
            file_tok = {k: v.to(device) for k, v in file_tok.items()}
            file_emb = model.encode_file(file_tok)  # [Bf, D]
            scores.extend((file_emb @ issue_emb).detach().cpu().tolist())

        ranked_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        ranked_paths = [cand_paths[j] for j in ranked_idx]

        mrr10_sum += mrr_at_k(ranked_paths, relevant, 10)
        r10_sum += recall_at_k(ranked_paths, relevant, 10)
        n += 1

    return {
        "mrr@10": mrr10_sum / n if n else 0.0,
        "recall@10": r10_sum / n if n else 0.0,
        "n": float(n),
        "candidate_n": float(candidate_n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--rerank_jsonl", type=str, required=True)
    ap.add_argument("--candidate_n", type=int, default=100)
    ap.add_argument("--max_len_issue", type=int, default=256)
    ap.add_argument("--max_len_file", type=int, default=256)
    ap.add_argument("--file_batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out_json", type=str, default="")

    args = ap.parse_args()

    metrics = eval_checkpoint(
        args.ckpt,
        args.rerank_jsonl,
        candidate_n=args.candidate_n,
        max_len_issue=args.max_len_issue,
        max_len_file=args.max_len_file,
        file_batch_size=args.file_batch_size,
        limit=args.limit,
    )
    print("Metrics:", metrics)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved:", args.out_json)


if __name__ == "__main__":
    main()
