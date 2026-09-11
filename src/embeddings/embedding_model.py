import hashlib
import re
from typing import List, Optional
import numpy as np

def hash_embed(texts: List[str], dimensions: int = 768) -> np.ndarray:
    """
    Local, deterministic bag-of-words hashing embedding using SHA-256.
    Returns normalized dense float32 array of shape (len(texts), dimensions).
    """
    if not texts:
        return np.zeros((0, dimensions), dtype=np.float32)

    embeddings = np.zeros((len(texts), dimensions), dtype=np.float32)
    for i, t in enumerate(texts):
        tokens = re.findall(r"\w+", t.lower())
        for token in tokens:
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions
            embeddings[i, idx] += 1.0

    # L2-normalization
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return embeddings / norms

# Alias for backward compatibility
embed = hash_embed

class EmbeddingModel:
    """Interface for text embeddings across providers (hashing, sentence-transformers, openai)."""
    def __init__(self, provider: str = "hashing", dimensions: int = 768, model_name: str = ""):
        self.provider = provider.lower()
        self.dimensions = dimensions
        self.model_name = model_name

    def embed_documents(self, texts: List[str], dimensions: Optional[int] = None) -> np.ndarray:
        d = dimensions if dimensions is not None else self.dimensions
        return hash_embed(texts, d)

    def embed_query(self, text: str, dimensions: Optional[int] = None) -> np.ndarray:
        d = dimensions if dimensions is not None else self.dimensions
        return hash_embed([text], d)
