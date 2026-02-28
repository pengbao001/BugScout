from __future__ import annotations

import random
from typing import Dict, List

from bugscout.data.examples import BugExample


def random_rank(
    examples: list[BugExample],
    candidates: dict[str, list[str]],
    *,
    seed: int = 999,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    preds: dict[str, list[str]] = {}
    for ex in examples:
        c = list(candidates[ex.example_id])
        rng.shuffle(c)
        preds[ex.example_id] = c
    return preds


def oracle_rank(
    examples: list[BugExample],
    candidates: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Oracle: put all relevant files first (perfect ranking).
    This should score extremely high and proves your metrics code is correct.
    """
    preds: dict[str, list[str]] = {}
    for ex in examples:
        rel = set(ex.changed_files)
        c = candidates[ex.example_id]
        positives_first = [x for x in c if x in rel]
        negatives_after = [x for x in c if x not in rel]
        preds[ex.example_id] = positives_first + negatives_after
    return preds