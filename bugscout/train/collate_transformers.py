from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class TokenizedBatch:
    issue : dict[str, torch.Tensor]
    file : dict[str, torch.Tensor]

def make_collate_fn(tokenizer : PreTrainedTokenizerBase, *, max_len_issue : int = 256, max_len_file : int = 256):

    def collate(items : list[dict[str, str]]) -> TokenizedBatch:
        issue_texts = [item["issue_text"] for item in items]
        file_texts = [item["file_text"] for item in items]

        issue_tok = tokenizer(
            issue_texts,
            padding=True,
            truncation=True,
            max_length=max_len_issue,
            return_tensors="pt",
        )

        file_tok = tokenizer(
            file_texts,
            padding=True,
            truncation=True,
            max_length=max_len_file,
            return_tensors="pt",
        )

        return TokenizedBatch(issue=dict(issue_tok), file=dict(file_tok))

    return collate