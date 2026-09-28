from typing import List, Tuple, Any, Callable

class DenseSimilaritySearch:
    """Performs cosine vector similarity search against ChromaDB."""
    def __init__(self, collection: Any, embed_fn: Callable[[List[str], int], Any], dimensions: int = 768):
        self.collection = collection
        self.embed_fn = embed_fn
        self.dimensions = dimensions

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Queries ChromaDB for nearest neighbors.
        Returns a list of (chunk_id, similarity_score) sorted by descending similarity.
        """
        if not self.collection:
            return []

        try:
            total_count = self.collection.count()
        except Exception:
            total_count = 0

        if total_count == 0:
            return []

        n_results = min(top_k, total_count)
        query_vec = self.embed_fn([query], self.dimensions).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=query_vec,
                n_results=n_results,
                include=["distances"]
            )
        except Exception:
            results = None

        candidates: List[Tuple[int, float]] = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            for chunk_id_str, dist in zip(ids, distances):
                # For cosine distance in [0, 2], cosine similarity is 1.0 - dist
                sim_score = max(0.0, 1.0 - float(dist))
                candidates.append((int(chunk_id_str), sim_score))

        return candidates
