import math

from bugscout.eval.metrics import ndcg_at_k, recall_at_k, mrr_at_k


def test_recall_at_k():
    ranked = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 10) == 0.5


def test_mrr_at_k():
    ranked = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert mrr_at_k(ranked, relevant, 1) == 0.0
    assert mrr_at_k(ranked, relevant, 3) == 1.0 / 2.0


def test_ndcg_at_k_binary():
    ranked = ["a", "b", "c"]
    relevant = {"b", "d"}

    # DCG@3: only "b" hits at rank 2
    dcg = 1.0 / math.log2(2 + 1)  # 1 / log2(3)
    # IDCG@3: two relevant items would be ranked at positions 1 and 2
    idcg = 1.0 / math.log2(1 + 1) + 1.0 / math.log2(2 + 1)

    expected = dcg / idcg
    got = ndcg_at_k(ranked, relevant, 3)
    assert abs(got - expected) < 1e-8


def test_duplicates_are_ignored():
    ranked = ["a", "b", "b", "c"]
    relevant = {"b"}
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert mrr_at_k(ranked, relevant, 10) == 1.0 / 2.0


def test_empty_relevant_returns_zero():
    ranked = ["a", "b"]
    relevant = set()
    assert recall_at_k(ranked, relevant, 10) == 0.0
    assert mrr_at_k(ranked, relevant, 10) == 0.0
    assert ndcg_at_k(ranked, relevant, 10) == 0.0