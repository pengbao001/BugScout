from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class HardNegBatch:
    issue: dict[str, torch.Tensor]      # [B, ...]
    files: dict[str, torch.Tensor]      # [(B*(1+N)), ...]
    num_negs: int                       # N


def make_hardneg_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_len_issue: int = 256,
    max_len_file: int = 256,
    num_negs: int = 20,
):
    """
    For each example, we create a list: [pos] + negs (length 1+N).
    We flatten all files across the batch so we can encode efficiently.
    """

    def collate(items: list[dict[str, Any]]) -> HardNegBatch:
        issue_texts = [it["issue_text"] for it in items]

        file_texts_flat: list[str] = []
        for it in items:
            negs = list(it["neg_texts"])[:num_negs]
            # Pad if fewer negs available (rare, but prevents crashes)
            while len(negs) < num_negs:
                negs.append(negs[-1] if negs else "__empty__")
            file_texts_flat.append(it["pos_text"])
            file_texts_flat.extend(negs)

        issue_tok = tokenizer(
            issue_texts,
            padding=True,
            truncation=True,
            max_length=max_len_issue,
            return_tensors="pt",
        )
        files_tok = tokenizer(
            file_texts_flat,
            padding=True,
            truncation=True,
            max_length=max_len_file,
            return_tensors="pt",
        )

        return HardNegBatch(issue=dict(issue_tok), files=dict(files_tok), num_negs=num_negs)

    return collate