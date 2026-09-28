import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Ingestion imports
from src.ingestion.text_cleaner import clean, clean_text
from src.ingestion.document_loader import load, load_document
from src.ingestion.ingestion_pipeline import IngestionPipeline

# Chunking imports
from src.chunking.fixed_chunking import split_fixed, split_boundary
from src.chunking.hierarchical_chunking import Chunk, split_hierarchical

# Embeddings imports
from src.embeddings.embedding_model import embed, hash_embed, EmbeddingModel

# Vector store imports
from src.vectorstore.chroma_db import ChromaStore

# Retrieval imports
from src.retrieval.similarity_search import DenseSimilaritySearch
from src.retrieval.retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import LexicalReranker

# Generation imports
from src.generation.prompt import format_context, format_retrieved_sources
from src.generation.llm import LLMGenerator

# Backward-compatibility alias
def split(t, size, overlap):
    return split_fixed(t, size, overlap)

# Load .env file
for line in Path(".env").read_text(encoding="utf-8").splitlines() if Path(".env").exists() else []:
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def env(k, d):
    return os.getenv(k, d)

@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(env("DATA_DIR", "data/raw"))
    index_dir: Path = Path(env("INDEX_DIR", "chroma_db"))
    chunk_size: int = int(env("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(env("CHUNK_OVERLAP", "150"))
    parent_chunk_size: int = int(env("PARENT_CHUNK_SIZE", "1800"))
    parent_chunk_overlap: int = int(env("PARENT_CHUNK_OVERLAP", "250"))
    top_k: int = int(env("TOP_K", "3"))
    dimensions: int = int(env("EMBEDDING_DIMENSIONS", "768"))
    collection: str = env("CHROMA_COLLECTION", "rag_documents")
    llm_provider: str = env("LLM_PROVIDER", "groq").lower()
    llm_model: str = env("LLM_MODEL", "openai/gpt-oss-20b")
    temperature: float = float(env("TEMPERATURE", "0.1"))
    reranker: str = env("RERANKER", "lexical").lower()

class RAGPipeline:
    """High-level facade orchestrating ingestion, embedding, storage, retrieval, and LLM answering."""
    def __init__(self):
        self.s = Settings()
        self.vector_store = ChromaStore(index_dir=self.s.index_dir, collection_name=self.s.collection)
        self.embed_model = EmbeddingModel(provider="hashing", dimensions=self.s.dimensions)
        self.llm_generator = LLMGenerator(
            provider=self.s.llm_provider,
            model=self.s.llm_model,
            temperature=self.s.temperature
        )
        self.query_rewriter = QueryRewriter(
            provider=self.s.llm_provider,
            model=self.s.llm_model,
            temperature=0.0
        )
        self.reranker = LexicalReranker() if self.s.reranker == "lexical" else None
        self.chunks: List[Chunk] = []
        self.retriever: Optional[HybridRetriever] = None

    def _init_retriever(self):
        if self.vector_store.collection is not None and self.chunks:
            dense_search = DenseSimilaritySearch(
                collection=self.vector_store.collection,
                embed_fn=self.embed_model.embed_documents,
                dimensions=self.s.dimensions
            )
            self.retriever = HybridRetriever(self.chunks, dense_search)

    def rebuild(self):
        """Processes raw files into hierarchical chunks, builds embeddings, and indexes in ChromaDB."""
        ingestion = IngestionPipeline(
            data_dir=self.s.data_dir,
            parent_chunk_size=self.s.parent_chunk_size,
            parent_chunk_overlap=self.s.parent_chunk_overlap,
            child_chunk_size=self.s.chunk_size,
            child_chunk_overlap=self.s.chunk_overlap,
        )
        self.chunks = ingestion.process()
        if not self.chunks:
            raise RuntimeError("Put documents in data/raw, then rebuild.")

        # Re-initialize collection in vector store
        self.vector_store.create_or_reset_collection(space="cosine")
        vectors = self.embed_model.embed_documents([c.search_text for c in self.chunks])
        self.vector_store.add_chunks(self.chunks, vectors, batch_size=100)

        # Cache chunks metadata
        chunks_dir = Path("data/chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        (chunks_dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks]),
            encoding="utf-8"
        )
        self._init_retriever()

    def load(self):
        """Loads cached chunks and connects to persistent ChromaDB collection."""
        chunks_path = Path("data/chunks/chunks.json")
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} not found. Run with --rebuild first.")
        self.chunks = [Chunk(**x) for x in json.loads(chunks_path.read_text(encoding="utf-8"))]
        self.vector_store.get_collection()
        self._init_retriever()

    def answer_structured(self, query: str) -> dict:
        """Retrieves relevant parent chunks via hybrid search (or fallback) and returns structured response dict."""
        if not self.retriever:
            if not self.chunks and Path("data/chunks/chunks.json").exists():
                self.load()
            else:
                self._init_retriever()

        # Clean and rewrite query for any grammatical errors or typos
        search_query = self.query_rewriter.rewrite(query) if self.query_rewriter else query
        is_rewritten = bool(search_query and search_query.lower() != query.lower().strip())

        # Retrieve candidates and apply reranker if enabled
        fetch_k = max(15, self.s.top_k * 5) if self.reranker else self.s.top_k
        found, mode = self.retriever.retrieve(search_query, top_k=fetch_k) if self.retriever else ([], "empty")
        
        if self.reranker and found:
            found = self.reranker.rerank(search_query, found, top_k=self.s.top_k)
        else:
            found = found[:self.s.top_k]

        context = format_context(found)
        answer = self.llm_generator.generate(search_query, context, found)

        return {
            "query": query,
            "search_query": search_query,
            "is_rewritten": is_rewritten,
            "retrieval_mode": mode.upper(),
            "answer": answer,
            "found_chunks": found,
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "filename": c.filename,
                    "page": c.page,
                    "text": c.text,
                }
                for c in found
            ],
        }

    def answer(self, query: str) -> str:
        """Retrieves relevant parent chunks via hybrid search (or fallback) and generates answer string."""
        res = self.answer_structured(query)
        rewritten_note = f"[Query Optimized: \"{res['search_query']}\"]\n\n" if res["is_rewritten"] else ""
        sources = format_retrieved_sources(res["found_chunks"])
        return f"{rewritten_note}[{res['retrieval_mode']}] Answer:\n{res['answer']}\n\nTop {len(res['sources'])} retrieved chunks:\n{sources}"

