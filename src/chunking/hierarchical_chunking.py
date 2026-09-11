from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.chunking.fixed_chunking import split_boundary

@dataclass
class Chunk:
    text: str              # Parent chunk context (used by LLM for answering)
    search_text: str       # Child chunk text (embedded and searched)
    source: str
    filename: str
    document_type: str
    page: Optional[int]
    chunk_id: int
    parent_id: int

def split_hierarchical(
    text: str,
    parent_size: int = 1800,
    parent_overlap: int = 250,
    child_size: int = 900,
    child_overlap: int = 150,
    source: str = "",
    filename: str = "",
    document_type: str = "",
    page: Optional[int] = None,
    start_chunk_id: int = 0,
    start_parent_id: int = 0,
) -> Tuple[List[Chunk], int, int]:
    """
    Splits text into parent chunks, then subdivides each parent into smaller child chunks.
    Returns:
        (chunks, next_chunk_id, next_parent_id)
    """
    if not text.strip():
        return [], start_chunk_id, start_parent_id

    chunks: List[Chunk] = []
    parent_chunks = split_boundary(text, parent_size, parent_overlap)
    
    current_chunk_id = start_chunk_id
    current_parent_id = start_parent_id
    
    for parent_text in parent_chunks:
        child_chunks = split_boundary(parent_text, child_size, child_overlap)
        # If parent was too small to split further, treat parent text as child text
        if not child_chunks:
            child_chunks = [parent_text]
            
        for child_text in child_chunks:
            chunks.append(
                Chunk(
                    text=parent_text,
                    search_text=child_text,
                    source=source,
                    filename=filename,
                    document_type=document_type,
                    page=page,
                    chunk_id=current_chunk_id,
                    parent_id=current_parent_id,
                )
            )
            current_chunk_id += 1
            
        current_parent_id += 1
        
    return chunks, current_chunk_id, current_parent_id
