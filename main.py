import argparse
from src.pipeline.rag_pipeline import RAGPipeline
if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--rebuild",action="store_true"); args=parser.parse_args()
    app=RAGPipeline(); args.rebuild and app.rebuild() or app.load()
    while True:
        try: q=input("Query (exit to quit): ").strip()
        except EOFError: break
        if q == "exit": break
        if q: print(app.answer(q))
