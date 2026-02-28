from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any, Iterable, Mapping, Sequence

def dedupe_preserve_order(items : Sequence[Any]) -> list[Any]:
    seen = set()
    out = []

    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

def to_ranked_items(ranking : Sequence[Any]) -> list[Any]:
    if 0 == len(ranking):
        return []
    
    first_item = ranking[0]
    if isinstance(first_item, (tuple, list)) and len(first_item) >= 1:
        return [x[0] for x in ranking]
    return list(ranking)

def recall_at_k(ranked_list : Sequence[Any], relevant : Iterable[Any], k : int) -> float:
    """
    Formula:
    Recall@K = (# relevant items retrieved in top K) / (total # relevant items)
    """

    uniq_set = set(relevant)
    if not uniq_set:
        return 0.0
    
    ranked = dedupe_preserve_order(to_ranked_items(ranked_list))
    topk = ranked[:max(k,0)]
    hits = sum(1 for x in topk if x in uniq_set)
    return hits / len(uniq_set)

def mrr_at_k(ranked_list : Sequence[Any], relevant : Iterable[Any], k : int) -> float:
    """
    Mean Reciprocal Rank@K for a single query:
    - Find the first relevant item in top K (1-indexed rank r)
    - Return 1/r
    - If none in top K, return 0.0
    """

    uniq_set = set(relevant)
    if not uniq_set:
        return 0.0
    
    ranked = dedupe_preserve_order(to_ranked_items(ranked_list))
    topk = ranked[:max(k,0)]
    
    for i, item in enumerate(topk, start=1):
        if item in uniq_set:
            return 1.0 / i
    return 0.0

def dcg_at_k(ranked_list : Sequence[Any], relevant : Iterable[Any], k : int) -> float:
    """
    DCG@K with binary relevance:
        DCG = sum_{i=1..k} uniq_set_i / log2(i+1)
        whiere uniq_set_i is 1 if item at rank i is relevant else 0.
    """
    uniq_set = set(relevant)
    if not uniq_set:
        return 0.0
    ranked = dedupe_preserve_order(to_ranked_items(ranked_list))
    topk = ranked[:max(k,0)]

    dcg = 0.0
    for i, item in enumerate(topk, start=1):
        uniq_set_i = 1.0 if item in uniq_set else 0.0
        dcg += uniq_set_i / log2(i + 1)
    
    return dcg

def ndcg_at_k(ranked_list : Sequence[Any], relevant : Iterable[Any], k : int) -> float:
    """
    nDCG@K = DCG@K / IDCG@K
    IDCG@K is DCG@K of an ideal ranking where all relevant items appear first.
    Uses binary relevance.
    """
    uniq_set = set(relevant)
    if not uniq_set:
        return 0.0
    
    dcg = dcg_at_k(ranked_list, uniq_set, k)

    ideal_hits = min(len(uniq_set), max(k,0))
    if 0 == ideal_hits:
        return 0.0
    
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / log2(i + 1)
    
    return 0.0 if idcg == 0.0 else (dcg/idcg)

@dataclass
class RankingMetrics:
    recall_at_1 : float
    recall_at_5 : float
    recall_at_10 : float
    mrr_at_10 : float
    ndcg_at_10 : float

def evaluate_ranking(ranked_list : Sequence[Any], relevant : Iterable[Any], *, ks : Sequence[int] = (1, 5, 10),) -> dict[str, float]:
    ks = list(ks)
    out = {}

    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked_list, relevant, k)
    
    out["mrr@10"] = mrr_at_k(ranked_list, relevant, 10)
    out["ndcg@10"] = ndcg_at_k(ranked_list, relevant, 10)

    return out

def evaluate_dataset(predictions : Mapping[str, Sequence[Any]], ground_truth : Mapping[str, Iterable[Any]], *, ks : Sequence[int] = (1, 5, 10),\
        skip_if_no_relevant : bool = True,) -> dict[str, float]:
    
    metric_sums = {}
    n = 0

    for qid, ranked in predictions.items():
        if qid not in ground_truth:
            continue

        targets = list(ground_truth[qid])
        if skip_if_no_relevant and len(targets) == 0:
            continue

        per = evaluate_ranking(ranked, targets, ks=ks)
        for k, v in per.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        n += 1
    
    if 0 == n:
        return {f"recall@{k}" : 0.0 for k in ks} | {"mrr@10" : 0.0, "ndcg@10" : 0.0}
    
    return {k : v / n for k,v in metric_sums.items()}


