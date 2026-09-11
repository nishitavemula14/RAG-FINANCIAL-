import csv
from pathlib import Path
from typing import Generator, Tuple, Optional

def load_document(path: Path) -> Generator[Tuple[str, Optional[int]], None, None]:
    """
    Loads text content from various file formats.
    Yields:
        (text_chunk, page_number_or_none)
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, 1):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    yield extracted, page_num
        except Exception as e:
            print(f"Warning: Failed to read PDF {path}: {e}")
            
    elif suffix == ".csv":
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                content = "\n".join(" | ".join(row) for row in csv.reader(f))
                if content.strip():
                    yield content, None
        except Exception as e:
            print(f"Warning: Failed to read CSV {path}: {e}")
            
    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            content = "\n".join(p.text for p in doc.paragraphs if p.text)
            if content.strip():
                yield content, None
        except Exception as e:
            print(f"Warning: Failed to read DOCX {path}: {e}")
            
    else:
        # Default text/markdown/json loader
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                yield content, None
        except Exception as e:
            print(f"Warning: Failed to read file {path}: {e}")

# Alias for backward compatibility
load = load_document
