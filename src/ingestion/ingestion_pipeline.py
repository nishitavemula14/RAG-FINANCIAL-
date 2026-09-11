from pathlib import Path
from typing import List
from src.ingestion.document_loader import load_document
from src.ingestion.text_cleaner import clean_text
from src.chunking.hierarchical_chunking import Chunk, split_hierarchical

class IngestionPipeline:
    """Orchestrates loading documents, cleaning text, and hierarchical chunking."""
    def __init__(
        self,
        data_dir: Path,
        parent_chunk_size: int = 1800,
        parent_chunk_overlap: int = 250,
        child_chunk_size: int = 900,
        child_chunk_overlap: int = 150,
    ):
        self.data_dir = Path(data_dir)
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

    def process(self) -> List[Chunk]:
        """Loads and chunks all files in data_dir."""
        chunks: List[Chunk] = []
        cid = 0
        pid = 0

        if not self.data_dir.exists():
            return chunks

        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                for text, page in load_document(path):
                    cleaned = clean_text(text)
                    if not cleaned:
                        continue
                    doc_chunks, cid, pid = split_hierarchical(
                        cleaned,
                        parent_size=self.parent_chunk_size,
                        parent_overlap=self.parent_chunk_overlap,
                        child_size=self.child_chunk_size,
                        child_overlap=self.child_chunk_overlap,
                        source=str(path),
                        filename=path.name,
                        document_type=path.suffix[1:].upper(),
                        page=page,
                        start_chunk_id=cid,
                        start_parent_id=pid,
                    )
                    chunks.extend(doc_chunks)

        return chunks
