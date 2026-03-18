# BugScout

BugScout is a repository-level bug localization system. Given a bug report and a repository snapshot, it ranks the source files most likely to be relevant to the fix.

## What it does

BugScout uses a two-stage pipeline:

1. **BM25 retrieval** to get a shortlist of candidate files
2. **Neural reranking** with a CodeBERT-based dual encoder to reorder the shortlist

This project was built as an end-to-end machine learning system:
- dataset construction
- repo checkout and caching
- BM25 baseline
- transformer training
- reranking evaluation
- ablation studies
- error analysis

## Project structure

```text
bugscout/
  data/
  models/
  retrieval/
  train/
cache/
checkpoints/
configs/
data/
results/
scripts/
test/
```
# Quick start
pip install -r requirements.txt
## Download and cache repositories
```code
$env:PYTHONPATH='.'
python scripts/day5_build_splits_jsonl.py \
  --configuration py \
  --hf_split dev \
  --limit 800 \
  --out_train data/train.jsonl \
  --out_val data/val.jsonl \
  --out_test data/test.jsonl
```
## Train
```code
$env:PYTHONPATH='.'
python scripts/train_dual_encoder.py \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --out_dir checkpoints/dual_encoder \
  --epochs 10 \
  --batch_size 8 \
  --eval_every_steps 1 \
  --num_workers 0
```
## Evaluate
```code
$env:PYTHONPATH='.'
python scripts/build_val_rerank_jsonl.py \
  --out data/val_rerank_title_body.jsonl \
  --title_only 0 \
  --include_path 1 \
  --topn 200
```