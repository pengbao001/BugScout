from __future__ import annotations

from torch.utils.data import DataLoader

from bugscout.train.jsonl_dataset import BugScoutJsonlDataset, collate_train_batch


def main() -> None:
    ds = BugScoutJsonlDataset("data/train_day5.jsonl")
    print("Dataset size:", len(ds))

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_train_batch)

    batch = next(iter(dl))
    print("\nBatch example:")
    print("issue_texts:", len(batch.issue_texts))
    print("pos_texts  :", len(batch.pos_texts))
    print("neg_texts  :", len(batch.neg_texts), "lists of negatives")
    print("neg per ex :", [len(x) for x in batch.neg_texts])

    # print a small snippet so you know it’s real
    print("\nIssue snippet:", batch.issue_texts[0][:120].replace("\n", " "))
    print("Pos snippet  :", batch.pos_texts[0][:120].replace("\n", " "))


if __name__ == "__main__":
    main()