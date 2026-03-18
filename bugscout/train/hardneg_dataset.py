from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class IssuePosNegJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, *, max_negs: int = 20) -> None:
        self.path = Path(jsonl_path)
        self.max_negs = max_negs
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
        negs = list(r.get("neg_texts", []))[: self.max_negs]
        return {
            "issue_text": r["issue_text"],
            "pos_text": r["pos_text"],
            "neg_texts": negs,
        }