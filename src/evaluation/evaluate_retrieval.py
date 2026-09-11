from typing import List, Dict, Any

def hit_rate_at_k(retrieved_ids: List[int], expected_id: int, k: int = 5) -> float:
    """Returns 1.0 if expected_id is in the top k retrieved IDs, else 0.0."""
    return 1.0 if expected_id in retrieved_ids[:k] else 0.0

def mean_reciprocal_rank(retrieved_rankings: List[List[int]], expected_ids: List[int]) -> float:
    """Calculates Mean Reciprocal Rank (MRR) across multiple queries."""
    if not retrieved_rankings or not expected_ids:
        return 0.0

    reciprocal_ranks = []
    for retrieved, expected in zip(retrieved_rankings, expected_ids):
        if expected in retrieved:
            rank = retrieved.index(expected) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def evaluate_retriever(retriever: Any, test_cases: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    """
    Runs evaluation against a list of test cases with schema:
    [{"query": "...", "expected_parent_id": 0}, ...]
    """
    hits = 0
    all_retrieved = []
    expected_ids = []

    for case in test_cases:
        query = case["query"]
        expected_id = case["expected_parent_id"]
        chunks, _ = retriever.retrieve(query, top_k=k)
        retrieved_parent_ids = [c.parent_id for c in chunks]

        hits += hit_rate_at_k(retrieved_parent_ids, expected_id, k=k)
        all_retrieved.append(retrieved_parent_ids)
        expected_ids.append(expected_id)

    total = len(test_cases)
    return {
        f"hit_rate@{k}": hits / total if total > 0 else 0.0,
        "mrr": mean_reciprocal_rank(all_retrieved, expected_ids),
        "total_queries": total,
    }
