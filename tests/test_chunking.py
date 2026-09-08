from src.pipeline.rag_pipeline import split
def test_overlap(): assert len(list(split("abcdefghij", 5, 1))) == 3
