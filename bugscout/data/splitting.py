from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset
import ast
import json
from typing import Iterable, Optional, Tuple
from bugscout.data.examples import BugExample
from collections import defaultdict
import random

def split_by_repo(examples : list[BugExample], *, train_ratio : float = 0.8, val_ratio : float = 0.1, seed : int = 66) -> tuple[list[BugExample], list[BugExample], list[BugExample]]:
    repo_to_examples : dict[str, list[BugExample]] = defaultdict(list)
    for example in examples:
        repo_to_examples[example.repo_id].append(example)
    
    repos = sorted(repo_to_examples.keys())
    rng = random.Random(seed)
    rng.shuffle(repos)

    n_repos = len(repos)
    n_train = int(train_ratio * n_repos)
    n_val = int(val_ratio * n_repos)

    train_repos = set(repos[:n_train])
    val_repos = set(repos[n_train : n_train + n_val])
    test_repos = set(repos[n_train + n_val : ])

    train, val, test = [], [], []

    for repo_id, exs in repo_to_examples.items():
        if repo_id in train_repos:
            train.extend(exs)
        elif repo_id in val_repos:
            val.extend(exs)
        else:
            test.extend(exs)
    
    return train, val, test

