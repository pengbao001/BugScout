from __future__ import annotations

from bugscout.data.lca_loader import load_lca_examples
from bugscout.data.splitting import split_by_repo
from bugscout.eval.candidates import build_candidates_for_examples, build_global_file_pool
from bugscout.eval.metrics import evaluate_dataset
from bugscout.eval.sanity_rankers import oracle_rank, random_rank


def main():
    # 1) Load a small subset so Day 2 runs fast
    examples = load_lca_examples(configuration="py", split="dev", limit=800)

    # 2) Split by repo to avoid leakage
    train, val, test = split_by_repo(examples, seed=42)

    print(f"Loaded: {len(examples)} examples")
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    # 3) For sanity check, evaluate on the test split
    # Build a candidate list per example, from a global file pool
    pool = build_global_file_pool(examples)
    test_candidates = build_candidates_for_examples(test, global_pool=pool, num_candidates=100)

    # 4) Ground truth mapping for evaluation
    ground_truth = {ex.example_id: ex.changed_files for ex in test}

    # 5) Random vs Oracle
    pred_random = random_rank(test, test_candidates)
    pred_oracle = oracle_rank(test, test_candidates)

    m_random = evaluate_dataset(pred_random, ground_truth, ks=(1, 5, 10))
    m_oracle = evaluate_dataset(pred_oracle, ground_truth, ks=(1, 5, 10))

    print("\n=== Random ranker ===")
    for k, v in sorted(m_random.items()):
        print(f"{k:10s} {v:.4f}")

    print("\n=== Oracle ranker ===")
    for k, v in sorted(m_oracle.items()):
        print(f"{k:10s} {v:.4f}")

    print("\nNote: Oracle recall@10 may be < 1.0 if some issues have >10 relevant files.")


if __name__ == "__main__":
    main()