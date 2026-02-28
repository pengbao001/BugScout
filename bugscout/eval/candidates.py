from __future__ import annotations

import random
from typing import Dict, List

from bugscout.data.examples import BugExample

def build_global_file_pool(examples : list[BugExample]) -> list[str]:
    pool = set()
    for example in examples:
        for f in example.changed_files:
            if f and isinstance(f, str):
                pool.add(f)
    
    return sorted(pool)

def build_candidates_for_examples(examples : list[BugExample], *, global_pool : list[str], num_candidates : int = 100, seed : int = 66) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out = {}

    for example in examples:
        pos = list(dict.fromkeys(example.changed_files))
        pos_set = set(pos)

        neg = [f for f in global_pool if f not in pos_set]
        rng.shuffle(neg)

        candidates = pos + neg
        candidates = candidates[ : max(num_candidates, len(pos))]

        out[example.example_id] = candidates
    
    return out

