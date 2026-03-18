from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_style_contrastive_loss(
issue_emb: torch.Tensor,  # [B, D]
    file_emb: torch.Tensor,   # [B, D]
    *,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes InfoNCE using in-batch negatives.

    logits = (issue_emb @ file_emb.T) / temperature
    labels = [0..B-1]
    """
    B = issue_emb.size(0)
    device = issue_emb.device

    logits = (issue_emb @ file_emb.T) / temperature  # [B, B]
    labels = torch.arange(B, device=device)

    loss_i2f = F.cross_entropy(logits, labels)

    if symmetric:
        loss_f2i = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_i2f + loss_f2i)
    else:
        loss = loss_i2f

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == labels).float().mean().item()
        top5 = (logits.topk(k=min(5, B), dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()

    stats = {
        "inbatch_top1": top1, 
        "inbatch_top5": top5
    }
    return loss, stats