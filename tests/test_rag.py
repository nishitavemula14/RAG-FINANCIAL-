from pathlib import Path
from src.ingestion.text_cleaner import clean_text
from src.chunking.semantic_chunking import split_semantic
from src.embeddings.embedding_model import EmbeddingModel, hash_embed
from src.generation.prompt import build_rag_prompt, format_context
from src.generation.llm import extractive_fallback
from src.evaluation.evaluate_retrieval import hit_rate_at_k, mean_reciprocal_rank
from src.evaluation.evaluate_answers import check_groundedness, has_citations, evaluate_answer
from src.chunking.hierarchical_chunking import Chunk

def test_text_cleaner():
    raw = "Hello \x00 world!   Multiple   spaces \n\n and newlines.  "
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "  " not in cleaned
    assert cleaned == "Hello world! Multiple spaces and newlines."

def test_semantic_chunking():
    text = "Sentence one is here. Sentence two is here. Sentence three is also here."
    chunks = split_semantic(text, target_size=45, overlap_sentences=1)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)

def test_embedding_model():
    model = EmbeddingModel(provider="hashing", dimensions=128)
    vecs = model.embed_documents(["Test document text", "Another query"])
    assert vecs.shape == (2, 128)
    q_vec = model.embed_query("Query")
    assert q_vec.shape == (1, 128)

def test_prompt_and_extractive_fallback():
    chunk = Chunk(
        text="Fiscal year 2020 revenue was 524 billion dollars.",
        search_text="Fiscal year 2020 revenue 524 billion",
        source="doc.txt",
        filename="doc.txt",
        document_type="TXT",
        page=1,
        chunk_id=0,
        parent_id=0
    )
    context = format_context([chunk])
    prompt = build_rag_prompt("What was fiscal 2020 revenue?", context)
    assert "Fiscal year 2020 revenue" in prompt

    answer = extractive_fallback("What was revenue in fiscal 2020?", [chunk])
    assert "524 billion" in answer

def test_evaluation_metrics():
    # Hit rate and MRR
    assert hit_rate_at_k([1, 2, 3, 4, 5], expected_id=3, k=3) == 1.0
    assert hit_rate_at_k([1, 2, 3, 4, 5], expected_id=4, k=3) == 0.0
    mrr = mean_reciprocal_rank([[1, 2], [3, 1]], expected_ids=[2, 3])
    assert mrr == (0.5 + 1.0) / 2

    # Answer evaluation
    ans = "Walmart reported strong revenue [1]."
    ctx = "Walmart reported strong revenue in fiscal 2020."
    assert has_citations(ans) is True
    eval_res = evaluate_answer("What was revenue?", ans, ctx)
    assert eval_res["has_citations"] is True
    assert eval_res["groundedness_score"] > 0.5
