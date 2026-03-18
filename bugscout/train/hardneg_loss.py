from __future__ import annotations

import torch
import torch.nn.functional as F


def hardneg_ce_loss(
    issue_emb: torch.Tensor,     # [B, D]
    file_emb: torch.Tensor,      # [B, 1+N, D] (pos at index 0)
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    logits[b, 0] = score(issue_b, pos_b)
    logits[b, i] = score(issue_b, neg_{b,i})
    label is always 0.
    """
    B, K, D = file_emb.shape
    assert issue_emb.shape == (B, D)

    # dot products: [B, K]
    logits = torch.einsum("bd,bkd->bk", issue_emb, file_emb) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == 0).float().mean().item()
        top5 = (logits.topk(k=min(5, K), dim=1).indices == 0).any(dim=1).float().mean().item()

    return loss, {"hardneg_top1": top1, "hardneg_top5": top5}