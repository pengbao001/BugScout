from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class TrainBatch:
    issue_texts: list[str]
    pos_texts: list[str]
    neg_texts: list[list[str]]  # per example list of negatives


class BugScoutJsonlDataset(Dataset):
    """
    Reads JSONL created by day5_build_train_jsonl.py.
    Each row already includes pos_text and neg_texts, so training is fast.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        return {
            "issue_text": r["issue_text"],
            "pos_text": r["pos_text"],
            "neg_texts": r["neg_texts"],
        }


def collate_train_batch(items: list[dict[str, Any]]) -> TrainBatch:
    """
    Simple collate: keep texts as lists of strings.
    Tokenization happens later during training.
    """
    issue_texts = [it["issue_text"] for it in items]
    pos_texts = [it["pos_text"] for it in items]
    neg_texts = [it["neg_texts"] for it in items]
    return TrainBatch(issue_texts=issue_texts, pos_texts=pos_texts, neg_texts=neg_texts)