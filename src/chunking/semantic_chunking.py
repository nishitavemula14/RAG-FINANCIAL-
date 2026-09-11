import re
from typing import List

def split_sentences(text: str) -> List[str]:
    """Splits text into sentences based on punctuation and whitespace."""
    if not text:
        return []
    # Split on sentence terminals followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]

def split_semantic(text: str, target_size: int = 800, overlap_sentences: int = 1) -> List[str]:
    """
    Groups complete sentences into semantically cohesive chunks up to target_size.
    Ensures complete thoughts are not broken across arbitrary character boundaries.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_length = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_length + sent_len > target_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Keep overlap_sentences for context continuity
            current_sentences = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_length = sum(len(s) + 1 for s in current_sentences)

        current_sentences.append(sent)
        current_length += sent_len + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
