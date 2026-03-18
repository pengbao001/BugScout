from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from bugscout.data.lca_loader import load_lca_examples
from bugscout.data.splitting import split_by_repo
from bugscout.data.repo_manager import RepoManager
from bugscout.data.candidates import collect_candidate_files
from bugscout.retrieval.bm25_ranker import BM25RepoIndex
from bugscout.retrieval.tokenize import normalize_rel_path
from bugscout.train.text_builders import build_file_input


def _dedupe_preserve(xs: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _choose_candidates_with_positive_coverage(
    all_candidates: list[str],
    positives_union: set[str],
    max_candidates: int,
) -> list[str]:
    """
    If repo has too many candidate files, cap it BUT ensure we keep all positives that exist.
    This avoids losing ground-truth files due to truncation.
    """
    if max_candidates <= 0 or len(all_candidates) <= max_candidates:
        return all_candidates

    positives_in_repo = [p for p in all_candidates if p in positives_union]
    positives_set = set(positives_in_repo)

    remaining = [p for p in all_candidates if p not in positives_set]
    kept = positives_in_repo + remaining
    return kept[:max_candidates]


def _sample_one(xs: list[str], rng: random.Random) -> str:
    return xs[rng.randrange(len(xs))]


def build_jsonl_for_split(
    examples,
    *,
    out_path: Path,
    split_name: str,
    mgr: RepoManager,
    rng: random.Random,
    bm25_mode: str,
    bm25_topn: int,
    num_hard_negs: int,
    num_rand_negs: int,
    max_candidates_per_repo: int,
    file_max_chars: int,
    max_repo_snapshots: int,
) -> None:
    """
    Build Day-5 style JSONL for one split (train/val/test).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by repo snapshot so we checkout once per (repo_id, base_sha)
    groups = defaultdict(list)
    for ex in examples:
        groups[(ex.repo_id, ex.base_sha)].append(ex)

    repo_items = list(groups.items())
    if max_repo_snapshots > 0:
        repo_items = repo_items[:max_repo_snapshots]

    written = 0
    skipped_git = 0
    skipped_no_pos = 0
    skipped_no_candidates = 0

    with out_path.open("w", encoding="utf-8") as f:
        for (repo_id, base_sha), exs in tqdm(repo_items, desc=f"{split_name}: repo snapshots"):
            # 1) Checkout repo snapshot
            try:
                repo_root = mgr.prepare_repo(repo_id, base_sha)
            except Exception as e:
                skipped_git += len(exs)
                print(f"\n[{split_name}] SKIP git error {repo_id}@{base_sha}: {e}")
                continue

            # 2) Candidate files
            file_records = collect_candidate_files(repo_root)
            all_candidates = [normalize_rel_path(r.rel_path) for r in file_records]
            all_candidates = _dedupe_preserve(all_candidates)

            if not all_candidates:
                skipped_no_candidates += len(exs)
                continue

            # 3) Compute union of positives in this repo snapshot (for safe capping)
            positives_union: set[str] = set()
            for ex in exs:
                for p in ex.changed_files:
                    positives_union.add(normalize_rel_path(p))

            # 4) Cap candidate count, but keep positives if possible
            file_paths = _choose_candidates_with_positive_coverage(
                all_candidates,
                positives_union=positives_union,
                max_candidates=max_candidates_per_repo,
            )
            candidate_set = set(file_paths)

            # 5) BM25 index for hard negative mining
            #    Start with bm25_mode="path" (fast). Later you can try "path+content".
            bm25 = BM25RepoIndex(
                repo_root=repo_root,
                file_paths=file_paths,
                mode=bm25_mode,            # "path", "content", "path+content"
                max_chars=file_max_chars,  # used when content is enabled
            )

            # 6) Build training records
            for ex in exs:
                positives = [normalize_rel_path(p) for p in ex.changed_files]
                positives = [p for p in positives if p in candidate_set]
                positives = _dedupe_preserve(positives)

                if not positives:
                    skipped_no_pos += 1
                    continue

                # Choose 1 positive per issue (simple and effective)
                pos_path = _sample_one(positives, rng)

                # Hard negatives: BM25 topN excluding positives
                hard_pool = bm25.rank(ex.issue_text, topk=bm25_topn)
                pos_set = set(positives)
                hard_pool = [p for p in hard_pool if p not in pos_set]
                hard_negs = hard_pool[:num_hard_negs]
                hard_neg_set = set(hard_negs)

                # Random negatives: sample from repo candidates excluding positives + hard negatives
                rand_pool = [p for p in file_paths if (p not in pos_set and p not in hard_neg_set)]
                rng.shuffle(rand_pool)
                rand_negs = rand_pool[:num_rand_negs]

                neg_paths = hard_negs + rand_negs

                # Build texts by reading only selected files
                pos_text = build_file_input(repo_root, pos_path, max_chars=file_max_chars)
                neg_texts = [build_file_input(repo_root, p, max_chars=file_max_chars) for p in neg_paths]

                record = {
                    "example_id": ex.example_id,
                    "repo_id": ex.repo_id,
                    "base_sha": ex.base_sha,
                    "issue_text": ex.issue_text,

                    "pos_path": pos_path,
                    "pos_text": pos_text,

                    "neg_paths": neg_paths,
                    "neg_texts": neg_texts,

                    "num_pos_total": len(positives),
                    "bm25_mode": bm25_mode,
                    "bm25_topn": bm25_topn,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"\n=== {split_name} split build finished ===")
    print("Output:", out_path)
    print("Written examples:", written)
    print("Skipped (git/checkout error):", skipped_git)
    print("Skipped (no candidates):", skipped_no_candidates)
    print("Skipped (no positives in candidates):", skipped_no_pos)


def main():
    ap = argparse.ArgumentParser()

    # Dataset loading
    ap.add_argument("--configuration", type=str, default="py")
    ap.add_argument("--hf_split", type=str, default="dev")
    ap.add_argument("--limit", type=int, default=800)

    # Output paths
    ap.add_argument("--out_train", type=str, default="data/train.jsonl")
    ap.add_argument("--out_val", type=str, default="data/val.jsonl")
    ap.add_argument("--out_test", type=str, default="data/test.jsonl")

    # Split ratios (repo-level)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=66)

    # Negative mining parameters
    ap.add_argument("--bm25_mode", type=str, choices=["path", "content", "path+content"], default="path")
    ap.add_argument("--bm25_topn", type=int, default=100)
    ap.add_argument("--num_hard_negs", type=int, default=10)
    ap.add_argument("--num_rand_negs", type=int, default=10)

    # Repo/candidate controls
    ap.add_argument("--max_candidates_per_repo", type=int, default=2000)
    ap.add_argument("--file_max_chars", type=int, default=8000)
    ap.add_argument("--cache_dir", type=str, default="cache")

    # For beginner fast iteration:
    ap.add_argument("--max_repo_snapshots", type=int, default=10,
                    help="If >0, only process this many (repo_id, base_sha) groups per split.")

    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) Load raw examples (small limit for iteration)
    examples = load_lca_examples(
        configuration=args.configuration,
        split=args.hf_split,
        limit=args.limit,
    )

    # 2) Split by repo_id to avoid leakage
    train_ex, val_ex, test_ex = split_by_repo(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Loaded: {len(examples)} examples from HF split={args.hf_split}")
    print(f"Repo-split sizes: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")

    # 3) Repo manager (uses caching)
    mgr = RepoManager(cache_dir=args.cache_dir)

    # 4) Build JSONL for each split
    build_jsonl_for_split(
        train_ex,
        out_path=Path(args.out_train),
        split_name="TRAIN",
        mgr=mgr,
        rng=rng,
        bm25_mode=args.bm25_mode,
        bm25_topn=args.bm25_topn,
        num_hard_negs=args.num_hard_negs,
        num_rand_negs=args.num_rand_negs,
        max_candidates_per_repo=args.max_candidates_per_repo,
        file_max_chars=args.file_max_chars,
        max_repo_snapshots=args.max_repo_snapshots,
    )

    build_jsonl_for_split(
        val_ex,
        out_path=Path(args.out_val),
        split_name="VAL",
        mgr=mgr,
        rng=rng,
        bm25_mode=args.bm25_mode,
        bm25_topn=args.bm25_topn,
        num_hard_negs=args.num_hard_negs,
        num_rand_negs=args.num_rand_negs,
        max_candidates_per_repo=args.max_candidates_per_repo,
        file_max_chars=args.file_max_chars,
        max_repo_snapshots=args.max_repo_snapshots,
    )

    build_jsonl_for_split(
        test_ex,
        out_path=Path(args.out_test),
        split_name="TEST",
        mgr=mgr,
        rng=rng,
        bm25_mode=args.bm25_mode,
        bm25_topn=args.bm25_topn,
        num_hard_negs=args.num_hard_negs,
        num_rand_negs=args.num_rand_negs,
        max_candidates_per_repo=args.max_candidates_per_repo,
        file_max_chars=args.file_max_chars,
        max_repo_snapshots=args.max_repo_snapshots,
    )


if __name__ == "__main__":
    main()