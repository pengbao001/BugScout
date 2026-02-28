from __future__ import annotations

from pathlib import Path

from bugscout.data.lca_loader import load_lca_examples
from bugscout.data.repo_manager import RepoManager
from bugscout.data.candidates import collect_candidate_files


def main() -> None:
    # Load a small set to pick one example (fast iteration)
    examples = load_lca_examples(configuration="py", split="dev", limit=20)
    ex = examples[0]

    print("Example:")
    print("  id      :", ex.example_id)
    print("  repo    :", ex.repo_id)
    print("  base_sha:", ex.base_sha)
    print("  #gt changed_files:", len(ex.changed_files))

    mgr = RepoManager(cache_dir="cache")

    # 1) Clone + checkout
    repo_path = mgr.prepare_repo(ex.repo_id, ex.base_sha)

    # 2) Candidate enumeration
    candidates = collect_candidate_files(Path(repo_path))
    cand_paths = {c.rel_path for c in candidates}

    # 3) Overlap sanity check: do GT files exist in candidates?
    gt = set(ex.changed_files)
    overlap = sorted([p for p in gt if p in cand_paths])

    print("\nRepo path:", repo_path)
    print("Candidate files:", len(candidates))
    print("Ground truth files:", len(gt))
    print("GT overlap with candidates:", len(overlap))

    if overlap:
        print("\nSome GT files found in candidates:")
        for p in overlap[:10]:
            print("  ", p)
    else:
        print("\nNo GT files were found in candidates.")
        print("This can happen if:")
        print("  - repo structure differs at base_sha")
        print("  - your filtering is too strict")
        print("  - some paths in changed_files are normalized differently")

    print("\nFirst 20 candidate files:")
    for r in candidates[:20]:
        print(f"  {r.rel_path} ({r.size_bytes} bytes)")


if __name__ == "__main__":
    main()