import re
from typing import List, Generator

def split_fixed(text: str, size: int, overlap: int) -> Generator[str, None, None]:
    """Basic sliding window chunker by character length."""
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece:
            yield piece
        if start + size >= len(text):
            break

def split_boundary(text: str, size: int, overlap: int) -> List[str]:
    """
    Boundary-aware chunker that attempts to split along paragraphs,
    sentences, or words before falling back to character slicing.
    """
    if not text:
        return []
    if len(text) <= size:
        return [text]
    
    # Try finding natural split points near size
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Look backwards from 'end' for natural boundary
            window = text[start:end]
            split_idx = -1
            for sep in ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]:
                idx = window.rfind(sep)
                # Keep split reasonable (at least half of chunk size if possible)
                if idx >= int(size * 0.4):
                    split_idx = start + idx + len(sep)
                    break
            if split_idx != -1 and split_idx > start:
                end = split_idx
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end >= len(text):
            break
            
        # Compute next start position with overlap
        start = max(start + 1, end - overlap)
        
    return chunks
