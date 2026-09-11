import re
from typing import List, Any, Tuple

class LexicalReranker:
    """Lightweight local reranker using token overlap and proximity density."""
    def __init__(self):
        pass

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        """Reranks chunks based on query term matches and phrase overlap."""
        if not chunks:
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))
        if not q_terms:
            return chunks[:top_k]

        scored: List[Tuple[float, Any]] = []
        for c in chunks:
            text_lower = c.text.lower()
            doc_terms = set(re.findall(r"\w+", text_lower))
            overlap_count = len(q_terms & doc_terms)
            
            # Phrase bonus: query substring appears in context
            phrase_bonus = 1.5 if query.lower() in text_lower else 0.0
            
            score = float(overlap_count) + phrase_bonus
            scored.append((score, c))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
