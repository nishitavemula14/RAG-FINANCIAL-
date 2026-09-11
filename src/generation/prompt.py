from typing import List, Any

SYSTEM_PROMPT = "You are a precise, source-grounded RAG assistant."

def format_context(chunks: List[Any]) -> str:
    """Formats retrieved parent chunks into numbered citation blocks."""
    return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))

def format_retrieved_sources(chunks: List[Any]) -> str:
    """Formats source chunks with filename and page metadata for user display."""
    return "\n\n".join(
        f"--- Chunk {i+1}: {c.filename}, page {c.page} ---\n{c.text}"
        for i, c in enumerate(chunks)
    )

def build_rag_prompt(question: str, context: str) -> str:
    """Constructs the prompt for context-grounded question answering."""
    return (
        f"Answer the question exactly and only from the context. "
        f"If the answer is not present, say so. Cite supporting chunk numbers like [1].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
