import re
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.retrieval.similarity_search import DenseSimilaritySearch

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

class HybridRetriever:
    """
    Hybrid Retriever that fuses Dense Vector Search (ChromaDB) and Sparse Search (BM25)
    using Reciprocal Rank Fusion (RRF).
    Falls back gracefully to pure dense vector similarity search when BM25 produces no matches.
    """
    def __init__(
        self,
        chunks: List[Any],
        dense_search: DenseSimilaritySearch,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.dense_search = dense_search
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.bm25: Optional[BM25Okapi] = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Builds BM25 index over child search_text."""
        if not self.chunks:
            return
        corpus = [tokenize(c.search_text) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Any], str]:
        """
        Retrieves top parent chunks using Hybrid Search or fallback to Pure Dense Vector Search.
        Returns:
            (parent_chunks, retrieval_mode) where retrieval_mode is 'hybrid' or 'fallback_dense_similarity'.
        """
        if not self.chunks:
            return [], "empty"

        # 1. Dense vector candidate retrieval (top 25 candidates)
        dense_candidates = self.dense_search.search(query, top_k=min(25, len(self.chunks)))
        dense_ranks: Dict[int, int] = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_candidates)}

        # 2. Check BM25 viability
        q_tokens = tokenize(query)
        bm25_active = False
        bm25_scores: List[float] = []

        if self.bm25 is not None and q_tokens:
            bm25_scores = self.bm25.get_scores(q_tokens)
            # Verify if there is any positive lexical match
            if any(score > 0.0 for score in bm25_scores):
                bm25_active = True

        # Fallback condition: If BM25 has no keyword overlap or is inactive,
        # fall back to pure dense vector similarity search.
        if not bm25_active or not dense_candidates:
            mode = "fallback_dense_similarity"
            selected_ids = [chunk_id for chunk_id, _ in dense_candidates]
        else:
            mode = "hybrid"
            # Get top BM25 ranked candidates
            ranked_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:25]
            bm25_ranks: Dict[int, int] = {idx: rank for rank, idx in enumerate(ranked_bm25_indices) if bm25_scores[idx] > 0}

            # 3. Reciprocal Rank Fusion (RRF)
            all_candidate_ids = set(dense_ranks.keys()).union(bm25_ranks.keys())
            rrf_scores: Dict[int, float] = {}

            for cid in all_candidate_ids:
                score = 0.0
                if cid in dense_ranks:
                    score += self.vector_weight / (self.rrf_k + dense_ranks[cid])
                if cid in bm25_ranks:
                    score += self.bm25_weight / (self.rrf_k + bm25_ranks[cid])
                rrf_scores[cid] = score

            selected_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # 4. Map child chunks to their unique parent chunks
        found_parents: List[Any] = []
        seen_parent_ids = set()

        for cid in selected_ids:
            if 0 <= cid < len(self.chunks):
                chunk = self.chunks[cid]
                if chunk.parent_id not in seen_parent_ids:
                    seen_parent_ids.add(chunk.parent_id)
                    found_parents.append(chunk)
                if len(found_parents) == top_k:
                    break

        return found_parents, mode
