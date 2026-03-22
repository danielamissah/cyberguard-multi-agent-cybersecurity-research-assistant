"""
FastAPI — CyberGuard Agent API
================================
REST API for the multi-agent cybersecurity research assistant.
"""

import time
import mlflow
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.graph.agent_graph import run_query, load_config, build_graph
from src.tools.knowledge_base import build_knowledge_base

app = FastAPI(
    title="CyberGuard — Multi-Agent Cybersecurity Research Assistant",
    description="5-agent LangGraph system for cybersecurity research",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()

# Warm up embedding model at startup so first query is fast
from src.graph.agent_graph import get_embedder
get_embedder(config)


# ── Request/Response models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    track: bool = True


class QueryResponse(BaseModel):
    query:         str
    response:      str
    quality_score: float
    sources:       list
    agent_trace:   list
    n_rag_results: int
    n_web_results: int
    latency_ms:    float
    timestamp:     str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def root():
    return {
        "name":        "CyberGuard Multi-Agent Research Assistant",
        "version":     "1.0.0",
        "agents":      ["Supervisor", "RAG", "WebSearch", "CodeAnalysis", "Critic"],
        "endpoints":   ["/health", "/query", "/query/batch", "/stats", "/graph/info"]
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()

    try:
        result = run_query(request.query, config)
        latency = (time.time() - start) * 1000

        # Log to MLflow if available
        if request.track:
            try:
                mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
                mlflow.set_experiment(config["mlflow"]["experiment"])
                with mlflow.start_run(run_name="agent_query"):
                    mlflow.log_metrics({
                        "quality_score": result["quality_score"],
                        "latency_ms":    latency,
                        "n_rag_results": result["n_rag_results"],
                        "n_web_results": result["n_web_results"],
                    })
            except Exception as mlflow_err:
                logger.warning(f"MLflow logging skipped: {mlflow_err}")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start) * 1000
    return QueryResponse(
        query=result["query"],
        response=result["response"],
        quality_score=result["quality_score"],
        sources=result["sources"],
        agent_trace=result["agent_trace"],
        n_rag_results=result["n_rag_results"],
        n_web_results=result["n_web_results"],
        latency_ms=round(latency, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/query/batch")
def batch_query(queries: list[str]):
    results = []
    for q in queries[:5]:  # cap at 5
        try:
            result = run_query(q, config)
            results.append(result)
        except Exception as e:
            results.append({"query": q, "error": str(e)})
    return results


@app.get("/stats")
def stats():
    try:
        import chromadb
        vs_cfg = config["vector_store"]
        client = chromadb.PersistentClient(path=vs_cfg["persist_dir"])
        col    = client.get_collection(vs_cfg["collection"])
        kb_count = col.count()
    except Exception:
        kb_count = 0

    return {
        "knowledge_base_chunks": kb_count,
        "agents":                5,
        "agent_names":           ["Supervisor", "RAG", "WebSearch", "CodeAnalysis", "Critic"],
        "llm_model":             config["llm"]["model"],
        "embedding_model":       config["vector_store"]["embedding_model"],
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    }


@app.get("/graph/info")
def graph_info():
    return {
        "nodes": [
            {"name": "supervisor",    "role": "Routes query to appropriate agents"},
            {"name": "rag",           "role": "Retrieves from cybersecurity knowledge base"},
            {"name": "web_search",    "role": "Searches live threat intelligence"},
            {"name": "code_analysis", "role": "Analyses malware and exploit patterns"},
            {"name": "synthesiser",   "role": "Combines all outputs into final response"},
            {"name": "critic",        "role": "Evaluates quality, triggers retry if needed"},
        ],
        "edges": [
            "supervisor → rag → web_search → code_analysis → synthesiser → critic",
            "critic → synthesiser (if quality < 0.7, max 3 retries)",
            "critic → END (if quality ≥ 0.7 or max iterations reached)",
        ],
        "framework": "LangGraph",
    }


@app.post("/knowledge-base/rebuild")
def rebuild_kb():
    try:
        build_knowledge_base(config, force_rebuild=True)
        return {"status": "rebuilt", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)