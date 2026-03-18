from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

class IssueFilePairJsonlDataset(Dataset):

    def __init__(self, jsonl_path : str | Path):
        self.path = Path(jsonl_path)
        self.rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx : int):
        r = self.rows[idx]
        return {
            "issue_text" : r["issue_text"],
            "file_text" : r["pos_text"],
        }