from src.pipeline.rag_pipeline import split
from src.chunking.fixed_chunking import split_boundary
from src.chunking.hierarchical_chunking import split_hierarchical

def test_overlap():
    assert len(list(split("abcdefghij", 5, 1))) == 3

def test_split_boundary():
    text = "Walmart Inc. reported strong revenue growth. Net sales reached record highs.\n\nOperating expenses decreased."
    chunks = split_boundary(text, size=50, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)

def test_hierarchical_chunking_parent_child_mapping():
    sample = "Paragraph 1 about quarterly financial results. " * 10 + "\n\n" + "Paragraph 2 about digital transformation. " * 10
    chunks, next_cid, next_pid = split_hierarchical(
        sample,
        parent_size=250,
        parent_overlap=50,
        child_size=100,
        child_overlap=20,
        filename="test.txt"
    )
    assert len(chunks) > 0
    assert next_cid == len(chunks)
    assert next_pid > 0
    # Verify every child chunk correctly references a valid parent_id
    for c in chunks:
        assert c.parent_id < next_pid
        # The parent context contains the child search text
        assert len(c.text) >= len(c.search_text)
