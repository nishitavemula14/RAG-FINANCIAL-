import React, { useState, useEffect } from "react";
import "./App.css";

const SAMPLE_QUERIES = [
  "How does Walmart distribute merchandise in its U.S. segment?",
  "What are Walmart's major operational risks?",
  "What was Walmart's net sales in fiscal 2020?",
  "How did Sam's Club segment perform?",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 4000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
      } else {
        setSystemHealth(null);
      }
    } catch {
      setSystemHealth(null);
    }
  };

  const handleSearch = async (searchQuery) => {
    const q = (typeof searchQuery === "string" ? searchQuery : query).trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setExpandedSources({});
    setCopied(false);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!res.ok) {
        if (res.status === 502 || res.status === 503 || res.status === 504) {
          throw new Error("Backend server is not running. Please open a terminal and run: python server.py");
        }
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with status ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSearch(query);
  };

  const toggleSource = (idx) => {
    setExpandedSources((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const handleCopyAnswer = () => {
    if (!result?.answer) return;
    navigator.clipboard.writeText(result.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="app-wrapper">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">🏛️</div>
          <div>
            <div className="brand-title">Walmart Financial RAG Assistant</div>
            <div className="brand-subtitle">Form 10-K Intelligent Query Engine</div>
          </div>
        </div>
        <div className="system-status">
          <div className={`status-dot ${systemHealth ? "online" : ""}`} />
          <span>
            {systemHealth
              ? `Connected (${systemHealth.total_chunks} chunks)`
              : "Checking backend..."}
          </span>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <h1>Ask Questions on Walmart's 10-K</h1>
        <p>
          Powered by Hierarchical Small-to-Big Chunking, BM25 + Cosine Hybrid Retrieval,
          and Lexical Reranking.
        </p>
      </section>

      {/* Quick Prompts */}
      <div className="quick-prompts">
        {SAMPLE_QUERIES.map((sample, idx) => (
          <button
            key={idx}
            type="button"
            className="prompt-chip"
            onClick={() => {
              setQuery(sample);
              handleSearch(sample);
            }}
          >
            {sample}
          </button>
        ))}
      </div>

      {/* Search Input Bar */}
      <form onSubmit={handleSubmit} className="search-card">
        <span style={{ fontSize: "18px", color: "var(--text-muted)" }}>🔍</span>
        <input
          type="text"
          className="search-input"
          placeholder="Ask a question (e.g., How does Walmart distribute merchandise in its U.S. segment?)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />
        {query && (
          <button
            type="button"
            className="clear-btn"
            onClick={() => setQuery("")}
            title="Clear input"
          >
            ✕
          </button>
        )}
        <button type="submit" className="submit-btn" disabled={loading || !query.trim()}>
          {loading ? (
            <>
              <div className="spinner" />
              <span>Searching...</span>
            </>
          ) : (
            <span>Search</span>
          )}
        </button>
      </form>

      {/* Error Alert */}
      {error && (
        <div className="error-card">
          <span style={{ fontSize: "20px" }}>⚠️</span>
          <div>
            <strong>Error: </strong> {error}
          </div>
        </div>
      )}

      {/* Optimization Banner */}
      {result?.is_rewritten && (
        <div className="optimized-banner">
          <span style={{ fontSize: "18px" }}>✨</span>
          <div>
            <strong>Query Rewritten for Precision: </strong>
            &ldquo;{result.search_query}&rdquo;
          </div>
        </div>
      )}

      {/* Answer Card */}
      {result && (
        <div className="answer-card">
          <div className="answer-header">
            <div className="answer-title">
              <span>💡 Generated Answer</span>
              <span className="badge-mode">{result.retrieval_mode} RETRIEVAL</span>
            </div>
            <button
              type="button"
              className="prompt-chip"
              style={{ padding: "4px 12px", fontSize: "12px" }}
              onClick={handleCopyAnswer}
            >
              {copied ? "✓ Copied" : "📋 Copy"}
            </button>
          </div>
          <div className="answer-body">{result.answer}</div>

          {/* Sources Section */}
          {result.sources && result.sources.length > 0 && (
            <div className="sources-section">
              <h3>
                <span>📚 Retrieved Source Chunks ({result.sources.length})</span>
              </h3>
              <div className="sources-list">
                {result.sources.map((src, idx) => {
                  const isExpanded = !!expandedSources[idx];
                  return (
                    <div key={idx} className="source-item">
                      <div className="source-header" onClick={() => toggleSource(idx)}>
                        <div className="source-meta">
                          <span className="source-badge">Chunk #{src.chunk_id}</span>
                          <span className="source-file">{src.filename}</span>
                          <span className="source-page">Page {src.page}</span>
                          <span className="source-page">(Parent #{src.parent_id})</span>
                        </div>
                        <span className="toggle-icon">
                          {isExpanded ? "▲ Hide Text" : "▼ View Source Context"}
                        </span>
                      </div>
                      {isExpanded && (
                        <div className="source-content">{src.text}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
