from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from bugscout.data.lca_loader import load_lca_examples
from bugscout.data.splitting import split_by_repo
from bugscout.data.repo_manager import RepoManager
from bugscout.data.candidates import collect_candidate_files
from bugscout.retrieval.bm25_ranker import BM25RepoIndex
from bugscout.retrieval.tokenize import normalize_rel_path
from bugscout.train.text_builders import build_file_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/val_rerank.jsonl")

    ap.add_argument("--configuration", type=str, default="py")
    ap.add_argument("--hf_split", type=str, default="dev")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--bm25_mode", type=str, choices=["path", "content", "path+content"], default="path")
    ap.add_argument("--topn", type=int, default=100)

    ap.add_argument("--max_candidates_per_repo", type=int, default=2000)
    ap.add_argument("--file_max_chars", type=int, default=8000)
    ap.add_argument("--cache_dir", type=str, default="cache")

    # Ablation flags (same meaning as Day5 builder)
    ap.add_argument("--title_only", type=int, default=0)
    ap.add_argument("--include_path", type=int, default=1)

    # for fast iteration
    ap.add_argument("--max_repo_snapshots", type=int, default=0)

    args = ap.parse_args()

    title_only = bool(args.title_only)
    include_path = bool(args.include_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_lca_examples(configuration=args.configuration, split=args.hf_split, limit=args.limit)
    train_ex, val_ex, test_ex = split_by_repo(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Loaded {len(examples)} examples. Using VAL split size={len(val_ex)}")

    # Group val by repo snapshot
    groups = defaultdict(list)
    for ex in val_ex:
        groups[(ex.repo_id, ex.base_sha)].append(ex)

    repo_items = list(groups.items())
    if args.max_repo_snapshots > 0:
        repo_items = repo_items[: args.max_repo_snapshots]

    mgr = RepoManager(cache_dir=args.cache_dir)

    written = 0
    skipped_git = 0
    skipped_no_relevant = 0

    with out_path.open("w", encoding="utf-8") as f:
        for (repo_id, base_sha), exs in tqdm(repo_items, desc="VAL repo snapshots"):
            try:
                repo_root = mgr.prepare_repo(repo_id, base_sha)
            except Exception as e:
                skipped_git += len(exs)
                print(f"\n[SKIP] git error {repo_id}@{base_sha}: {e}")
                continue

            file_records = collect_candidate_files(repo_root)
            file_paths = [normalize_rel_path(r.rel_path) for r in file_records]
            if len(file_paths) > args.max_candidates_per_repo:
                file_paths = file_paths[: args.max_candidates_per_repo]
            candidate_set = set(file_paths)

            bm25 = BM25RepoIndex(
                repo_root=repo_root,
                file_paths=file_paths,
                mode=args.bm25_mode,
                max_chars=args.file_max_chars,
            )

            for ex in exs:
                issue_text = ex.issue_text
                if title_only:
                    issue_text = issue_text.split("\n\n", 1)[0]

                relevant = [normalize_rel_path(p) for p in ex.changed_files]
                relevant = [p for p in relevant if p in candidate_set]

                if not relevant:
                    skipped_no_relevant += 1
                    continue

                cand_paths = bm25.rank(issue_text, topk=args.topn)
                cand_texts = [
                    build_file_input(repo_root, p, max_chars=args.file_max_chars, include_path=include_path)
                    for p in cand_paths
                ]

                rec = {
                    "example_id": ex.example_id,
                    "repo_id": ex.repo_id,
                    "base_sha": ex.base_sha,
                    "issue_text": issue_text,
                    "candidate_paths": cand_paths,
                    "candidate_texts": cand_texts,
                    "relevant_paths": relevant,
                    "title_only": title_only,
                    "include_path": include_path,
                    "bm25_mode": args.bm25_mode,
                    "topn": args.topn,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print("\n=== VAL rerank JSONL built ===")
    print("Output:", out_path)
    print("Written:", written)
    print("Skipped git:", skipped_git)
    print("Skipped no relevant:", skipped_no_relevant)


if __name__ == "__main__":
    main()
