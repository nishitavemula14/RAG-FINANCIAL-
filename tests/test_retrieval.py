from unittest.mock import MagicMock
from src.chunking.hierarchical_chunking import Chunk
from src.retrieval.similarity_search import DenseSimilaritySearch
from src.retrieval.retriever import HybridRetriever

def create_sample_chunks():
    return [
        Chunk(
            text="Parent chunk 0: Walmart net sales reached 524 billion dollars in fiscal 2020.",
            search_text="Walmart net sales 524 billion fiscal 2020",
            source="walmart_2020.pdf",
            filename="walmart_2020.pdf",
            document_type="PDF",
            page=1,
            chunk_id=0,
            parent_id=0,
        ),
        Chunk(
            text="Parent chunk 0: Walmart net sales reached 524 billion dollars in fiscal 2020.",
            search_text="eCommerce sales increased 37 percent globally",
            source="walmart_2020.pdf",
            filename="walmart_2020.pdf",
            document_type="PDF",
            page=1,
            chunk_id=1,
            parent_id=0,
        ),
        Chunk(
            text="Parent chunk 1: Risk factors include supply chain disruptions and competition.",
            search_text="Risk factors supply chain disruptions and retail competition",
            source="walmart_2020.pdf",
            filename="walmart_2020.pdf",
            document_type="PDF",
            page=2,
            chunk_id=2,
            parent_id=1,
        ),
    ]

def test_hybrid_retrieval_with_keyword_match():
    chunks = create_sample_chunks()
    # Mock dense search to return candidate list
    mock_dense = MagicMock(spec=DenseSimilaritySearch)
    mock_dense.search.return_value = [(0, 0.95), (1, 0.80), (2, 0.40)]

    retriever = HybridRetriever(chunks, mock_dense)
    results, mode = retriever.retrieve("What were net sales in fiscal 2020?", top_k=2)

    assert mode == "hybrid"
    assert len(results) >= 1
    # Check parent deduplication: chunk 0 and 1 share parent_id 0, so only one parent 0 should be present
    parent_ids = [c.parent_id for c in results]
    assert len(parent_ids) == len(set(parent_ids))
    assert results[0].parent_id == 0

def test_fallback_to_pure_dense_search_on_zero_keyword_match():
    chunks = create_sample_chunks()
    mock_dense = MagicMock(spec=DenseSimilaritySearch)
    mock_dense.search.return_value = [(2, 0.85), (0, 0.50)]

    retriever = HybridRetriever(chunks, mock_dense)
    # Query with out-of-vocabulary terms not present in chunks
    results, mode = retriever.retrieve("xyzabc qwerty entirelyunrelatedterms", top_k=2)

    # BM25 scores will be 0 for all chunks, triggering automatic fallback to pure dense vector search
    assert mode == "fallback_dense_similarity"
    assert len(results) == 2
    assert results[0].parent_id == 1
