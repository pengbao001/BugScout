from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


Pooling = Literal["cls", "mean"]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: [B, T, H]
    attention_mask: [B, T] (1 for real tokens, 0 for padding)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)                   # [B, H]
    denom = mask.sum(dim=1).clamp(min=1.0)                     # [B, 1]
    return summed / denom


@dataclass
class DualEncoderConfig:
    model_name: str
    pooling: Pooling = "mean"
    share_weights: bool = True
    proj_dim: Optional[int] = None   # e.g. 256 to reduce size, or None to keep hidden size
    normalize: bool = True           # cosine similarity usually helps


class DualEncoder(nn.Module):
    def __init__(self, cfg: DualEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.issue_encoder = AutoModel.from_pretrained(cfg.model_name)

        if cfg.share_weights:
            self.file_encoder = self.issue_encoder
        else:
            self.file_encoder = AutoModel.from_pretrained(cfg.model_name)

        hidden_size = self.issue_encoder.config.hidden_size

        if cfg.proj_dim is not None:
            self.issue_proj = nn.Linear(hidden_size, cfg.proj_dim, bias=False)
            self.file_proj = nn.Linear(hidden_size, cfg.proj_dim, bias=False)
            out_dim = cfg.proj_dim
        else:
            self.issue_proj = None
            self.file_proj = None
            out_dim = hidden_size

        self.out_dim = out_dim

    def encode(self, encoder: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = encoder(**batch)
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        if self.cfg.pooling == "cls":
            emb = last_hidden[:, 0]  # [B, H]
        else:
            emb = mean_pool(last_hidden, batch["attention_mask"])

        return emb

    def encode_issue(self, issue_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode issue texts -> [B, D]
        Applies pooling + optional projection + optional normalization.
        """
        issue_emb = self.encode(self.issue_encoder, issue_batch)

        if self.issue_proj is not None:
            issue_emb = self.issue_proj(issue_emb)

        if self.cfg.normalize:
            issue_emb = F.normalize(issue_emb, p=2, dim=-1)

        return issue_emb

    def encode_file(self, file_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode file texts -> [B, D]
        Applies pooling + optional projection + optional normalization.
        """
        file_emb = self.encode(self.file_encoder, file_batch)

        if self.file_proj is not None:
            file_emb = self.file_proj(file_emb)

        if self.cfg.normalize:
            file_emb = F.normalize(file_emb, p=2, dim=-1)

        return file_emb

    def forward(
        self,
        issue_batch: dict[str, torch.Tensor],
        file_batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        issue_emb = self.encode_issue(issue_batch)
        file_emb = self.encode_file(file_batch)
        return issue_emb, file_emb
