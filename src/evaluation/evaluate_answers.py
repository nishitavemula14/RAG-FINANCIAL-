import re
from typing import Dict, Any

def has_citations(answer: str) -> bool:
    """Checks whether the generated answer contains bracketed citations (e.g. [1], [2])."""
    return bool(re.search(r"\[\d+\]", answer))

def check_groundedness(answer: str, context: str) -> float:
    """
    Computes lexical groundedness score (ratio of answer terms found in the retrieved context).
    Higher ratio indicates stronger fidelity to the provided source.
    """
    answer_terms = set(re.findall(r"\w+", answer.lower()))
    # Remove common stop words for a more meaningful overlap check
    stop_words = {"the", "a", "an", "is", "in", "it", "of", "to", "and", "or", "for", "on", "was", "were", "are"}
    meaningful_terms = answer_terms - stop_words
    if not meaningful_terms:
        return 1.0

    context_terms = set(re.findall(r"\w+", context.lower()))
    overlap = meaningful_terms & context_terms
    return len(overlap) / len(meaningful_terms)

def evaluate_answer(question: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Evaluates generated answer for citation presence, length, and groundedness in context.
    """
    groundedness = check_groundedness(answer, context)
    cited = has_citations(answer)
    refused = "not present" in answer.lower() or "no relevant information" in answer.lower()

    return {
        "groundedness_score": round(groundedness, 4),
        "has_citations": cited,
        "is_refusal": refused,
        "answer_length": len(answer),
    }
