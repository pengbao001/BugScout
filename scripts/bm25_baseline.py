from __future__ import annotations

from collections import defaultdict

from bugscout.data.lca_loader import load_lca_examples
from bugscout.data.repo_manager import RepoManager
from bugscout.data.candidates import collect_candidate_files
from bugscout.data.splitting import split_by_repo
from bugscout.eval.metrics import evaluate_dataset
from bugscout.retrieval.bm25_ranker import BM25RepoIndex
from bugscout.retrieval.tokenize import normalize_rel_path


def build_ground_truth_covered(ex, candidate_set: set[str]) -> list[str]:
    """
    Only count ground-truth files that actually exist in our candidate set.
    This isolates BM25 quality from candidate-generation mismatches.
    """
    gt = [normalize_rel_path(p) for p in ex.changed_files]
    gt_covered = [p for p in gt if p in candidate_set]
    return gt_covered


def main() -> None:
    # Start small for speed (increase later)
    examples = load_lca_examples(configuration="py", split="dev", limit=800)

    train, val, test = split_by_repo(examples, seed=42)
    print(f"Loaded: {len(examples)} examples")
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    # Group by repo+sha so we don’t rebuild indexes repeatedly
    groups = defaultdict(list)
    for ex in test:
        groups[(ex.repo_id, ex.base_sha)].append(ex)

    mgr = RepoManager(cache_dir="cache")

    # We'll evaluate 3 BM25 modes
    modes = ["path", "content", "path+content"]

    # predictions[mode][example_id] = ranked_list
    predictions = {m: {} for m in modes}
    ground_truth = {}  # {example_id: covered_relevant_files}
    total = 0
    covered = 0

    for (repo_id, base_sha), exs in groups.items():
        print(f"\n[Repo] {repo_id} @ {base_sha}  (issues={len(exs)})")

        # 1) Prepare repo at commit
        repo_path = mgr.prepare_repo(repo_id, base_sha)

        # 2) Enumerate candidate files
        file_records = collect_candidate_files(repo_path)
        file_paths = [r.rel_path for r in file_records]
        candidate_set = set(file_paths)

        print(f"  Candidates: {len(file_paths)}")

        # 3) Build BM25 indexes for each mode (baseline variants)
        indexes = {
            m: BM25RepoIndex(repo_root=repo_path, file_paths=file_paths, mode=m, max_chars=8000)
            for m in modes
        }

        # 4) Rank each issue in this repo snapshot
        for ex in exs:
            total += 1
            gt_covered = build_ground_truth_covered(ex, candidate_set)
            ground_truth[ex.example_id] = gt_covered

            if len(gt_covered) > 0:
                covered += 1

            for m in modes:
                ranked = indexes[m].rank(ex.issue_text, topk=200)
                predictions[m][ex.example_id] = ranked

    coverage = covered / total if total > 0 else 0.0
    print(f"\nCoverage (issues with >=1 GT file in candidates): {covered}/{total} = {coverage:.3f}")

    # Evaluate (skip examples where gt_covered is empty)
    # This focuses evaluation on ranking quality (not candidate-generation mismatch).
    for m in modes:
        metrics = evaluate_dataset(
            predictions[m],
            ground_truth,
            ks=(1, 5, 10),
            skip_if_no_relevant=True,
        )
        print(f"\n=== BM25 baseline: {m} ===")
        for k, v in sorted(metrics.items()):
            print(f"{k:10s} {v:.4f}")


if __name__ == "__main__":
    main()