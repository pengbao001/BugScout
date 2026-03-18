from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from bugscout.models.dual_encoder import DualEncoder, DualEncoderConfig
from bugscout.train.pairs_dataset import IssueFilePairJsonlDataset
from bugscout.train.collate_transformers import make_collate_fn
from bugscout.train.contrastive import clip_style_contrastive_loss


@torch.no_grad()
def evaluate(model: DualEncoder, dl: DataLoader, device: torch.device, temperature: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n = 0

    for batch in dl:
        issue = {k: v.to(device) for k, v in batch.issue.items()}
        file = {k: v.to(device) for k, v in batch.file.items()}

        issue_emb, file_emb = model(issue, file)
        loss, stats = clip_style_contrastive_loss(issue_emb, file_emb, temperature=temperature, symmetric=True)

        bs = issue_emb.size(0)
        total_loss += loss.item() * bs
        total_top1 += stats["inbatch_top1"] * bs
        total_top5 += stats["inbatch_top5"] * bs
        n += bs

    if n == 0:
        return {"val_loss": 0.0, "val_inbatch_top1": 0.0, "val_inbatch_top5": 0.0}

    return {
        "val_loss": total_loss / n,
        "val_inbatch_top1": total_top1 / n,
        "val_inbatch_top5": total_top5 / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default="data/train.jsonl")
    ap.add_argument("--val_jsonl", type=str, default="data/val.jsonl")
    ap.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    ap.add_argument("--pooling", type=str, choices=["cls", "mean"], default="mean")
    ap.add_argument("--share_weights", action="store_true")
    ap.add_argument("--no_share_weights", dest="share_weights", action="store_false")
    ap.set_defaults(share_weights=True)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_len_issue", type=int, default=256)
    ap.add_argument("--max_len_file", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=(0 if os.name == "nt" else 2))

    ap.add_argument("--out_dir", type=str, default="checkpoints/dual_encoder")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collate_fn = make_collate_fn(
        tokenizer,
        max_len_issue=args.max_len_issue,
        max_len_file=args.max_len_file,
    )

    # Datasets & loaders
    train_ds = IssueFilePairJsonlDataset(args.train_jsonl)
    val_ds = IssueFilePairJsonlDataset(args.val_jsonl)
    pin = (device.type == "cuda")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # Model
    cfg = DualEncoderConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        share_weights=args.share_weights,
        proj_dim=None,
        normalize=True,
    )
    model = DualEncoder(cfg).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_dl)
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Mixed precision
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("Train examples:", len(train_ds))
    print("Val examples  :", len(val_ds))
    print("Total steps   :", total_steps, "Warmup:", warmup_steps)
    print("Config:", asdict(cfg))

    best_val_loss = float("inf")
    global_step = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            global_step += 1

            issue = {k: v.to(device, non_blocking=True) for k, v in batch.issue.items()}
            file = {k: v.to(device, non_blocking=True) for k, v in batch.file.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                issue_emb, file_emb = model(issue, file)
                loss, stats = clip_style_contrastive_loss(
                    issue_emb, file_emb,
                    temperature=args.temperature,
                    symmetric=True,
                )

            scaler.scale(loss).backward()

            # Gradient clipping (unscale first!)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                top1=f"{stats['inbatch_top1']:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Periodic validation
            if global_step % args.eval_every_steps == 0:
                val_metrics = evaluate(model, val_dl, device, temperature=args.temperature)
                print(f"\n[Step {global_step}] val:", val_metrics)

                # Save best checkpoint by val_loss
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    ckpt_path = out_dir / "best.pt"
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "cfg": asdict(cfg),
                            "tokenizer_name": args.model_name,
                            "step": global_step,
                            "val_metrics": val_metrics,
                        },
                        ckpt_path,
                    )
                    print(f"Saved new best checkpoint -> {ckpt_path}")

                model.train()

        # End-of-epoch validation too
        val_metrics = evaluate(model, val_dl, device, temperature=args.temperature)
        print(f"\n[Epoch {epoch}] val:", val_metrics)

    print("\nTraining complete. Best val_loss:", best_val_loss)


if __name__ == "__main__":
    main()