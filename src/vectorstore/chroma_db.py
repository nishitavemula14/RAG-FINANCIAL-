from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
import numpy as np

class ChromaStore:
    """Manages persistent ChromaDB vector storage and indexing."""
    def __init__(self, index_dir: Path, collection_name: str = "rag_documents"):
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.index_dir))
        self.collection = None

    def create_or_reset_collection(self, space: str = "cosine"):
        """Deletes existing collection if present and creates a fresh one."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": space}
        )
        return self.collection

    def get_collection(self):
        """Loads an existing collection."""
        self.collection = self.client.get_collection(self.collection_name)
        return self.collection

    def add_chunks(self, chunks: List[Any], embeddings: np.ndarray, batch_size: int = 100):
        """Batches and adds chunks and their embeddings into ChromaDB."""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_or_reset_collection or get_collection first.")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_vectors = embeddings[i:i + batch_size].tolist()
            self.collection.add(
                ids=[str(c.chunk_id) for c in batch],
                documents=[c.search_text for c in batch],
                embeddings=batch_vectors
            )

    def count(self) -> int:
        return self.collection.count() if self.collection else 0
