import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.pipeline.rag_pipeline import RAGPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

logger.info("Initializing RAG Pipeline...")
rag = RAGPipeline()
try:
    rag.load()
    logger.info(f"RAG Pipeline successfully loaded with {len(rag.chunks)} chunks.")
except Exception as e:
    logger.warning(f"Could not load pre-indexed chunks: {e}. Run with --rebuild first.")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "total_chunks": len(rag.chunks),
        "llm_provider": rag.s.llm_provider,
        "llm_model": rag.s.llm_model,
        "top_k": rag.s.top_k,
    })

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True, silent=True) or {}
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    logger.info(f"Received query: {user_query}")
    try:
        result = rag.answer_structured(user_query)
        # Remove raw Chunk objects before JSON serialization
        result.pop("found_chunks", None)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error processing query:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f" RAG Backend Server running at: http://localhost:{port}")
    print(f" API Endpoints: /api/health , /api/query")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

