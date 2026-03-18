from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from bugscout.models.dual_encoder import DualEncoder, DualEncoderConfig
from bugscout.train.hardneg_dataset import IssuePosNegJsonlDataset
from bugscout.train.collate_hardneg import make_hardneg_collate_fn
from bugscout.train.hardneg_loss import hardneg_ce_loss
from bugscout.eval.rerank_eval import load_rerank_jsonl, evaluate_rerank_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default="data/train_hard.jsonl")
    ap.add_argument("--val_rerank_jsonl", type=str, default="data/val_rerank.jsonl")

    ap.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    ap.add_argument("--pooling", type=str, choices=["cls", "mean"], default="mean")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--max_len_issue", type=int, default=256)
    ap.add_argument("--max_len_file", type=int, default=256)
    ap.add_argument("--num_negs", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every_steps", type=int, default=1)
    ap.add_argument("--val_limit", type=int, default=200)  # fast iteration
    ap.add_argument("--file_batch_size", type=int, default=32)

    ap.add_argument("--out_dir", type=str, default="checkpoints\hard_only")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Training data (subset-friendly)
    pin = (device.type == "cuda")
    train_ds = IssuePosNegJsonlDataset(args.train_jsonl, max_negs=args.num_negs)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_hardneg_collate_fn(
            tokenizer,
            max_len_issue=args.max_len_issue,
            max_len_file=args.max_len_file,
            num_negs=args.num_negs,
        ),
        num_workers=0,
        pin_memory=pin,
    )

    # Model
    cfg = DualEncoderConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        share_weights=True,
        proj_dim=None,
        normalize=True,
    )
    model = DualEncoder(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Preload val rerank set (fast eval)
    val_rows = load_rerank_jsonl(args.val_rerank_jsonl, limit=args.val_limit)
    print("Train rows:", len(train_ds))
    print("Val rerank rows:", len(val_rows))
    print("Total steps:", total_steps)

    best_mrr10 = -1.0
    global_step = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            global_step += 1

            issue = {k: v.to(device, non_blocking=True) for k, v in batch.issue.items()}
            files = {k: v.to(device, non_blocking=True) for k, v in batch.files.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                issue_emb, file_emb_flat = model(issue, files)  # file_emb_flat: [(B*(1+N)), D]
                B = issue_emb.size(0)
                K = 1 + batch.num_negs
                file_emb = file_emb_flat.view(B, K, -1)        # [B, 1+N, D]
                loss, stats = hardneg_ce_loss(issue_emb, file_emb, temperature=args.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", top1=f"{stats['hardneg_top1']:.3f}")

            # Periodic validation using real metric
            if global_step % args.eval_every_steps == 0:
                metrics = evaluate_rerank_dataset(
                    model,
                    tokenizer,
                    val_rows,
                    device=device,
                    max_len_issue=args.max_len_issue,
                    max_len_file=args.max_len_file,
                    file_batch_size=args.file_batch_size,
                )
                print(f"\n[Step {global_step}] val metrics:", metrics)

                if metrics["mrr@10"] > best_mrr10:
                    best_mrr10 = metrics["mrr@10"]
                    ckpt_path = out_dir / "best.pt"
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "cfg": asdict(cfg),
                            "tokenizer_name": args.model_name,
                            "step": global_step,
                            "val_metrics": metrics,
                        },
                        ckpt_path,
                    )
                    print("Saved best checkpoint:", ckpt_path)

                model.train()

    print("\nTraining complete. Best val mrr@10:", best_mrr10)


if __name__ == "__main__":
    main()