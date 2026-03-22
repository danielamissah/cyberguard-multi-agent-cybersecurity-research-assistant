"""
Knowledge Base Builder
======================
Fetches cybersecurity research papers from arXiv,
chunks them, and indexes into ChromaDB for RAG retrieval.
"""

import os
import time
import yaml
import arxiv
import chromadb
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_arxiv_papers(queries: list, max_papers: int = 200) -> list:
    """Fetch cybersecurity papers from arXiv."""
    client = arxiv.Client()
    papers = []
    per_query = max(1, max_papers // len(queries))

    for query in queries:
        logger.info(f"Fetching: {query}")
        try:
            search = arxiv.Search(
                query=query,
                max_results=per_query,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for paper in client.results(search):
                papers.append({
                    "id":       paper.entry_id,
                    "title":    paper.title,
                    "abstract": paper.summary,
                    "authors":  ", ".join(str(a) for a in paper.authors[:3]),
                    "year":     paper.published.year,
                    "url":      paper.entry_id,
                    "query":    query,
                })
            time.sleep(3)  # respect arXiv rate limits
        except Exception as e:
            logger.warning(f"Failed to fetch '{query}': {e}")

    # Deduplicate by ID
    seen = set()
    unique = []
    for p in papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)

    logger.success(f"Fetched {len(unique)} unique papers")
    return unique


def chunk_papers(papers: list, chunk_size: int = 512,
                 overlap: int = 64) -> list:
    """Split paper title+abstract into overlapping chunks."""
    chunks = []
    for paper in papers:
        # Combine title and abstract for richer context
        text = (
            f"Title: {paper['title']}\n"
            f"Authors: {paper['authors']}\n"
            f"Year: {paper['year']}\n"
            f"Abstract: {paper['abstract']}"
        )
        words = text.split()

        # Always produce at least one chunk
        if len(words) <= chunk_size:
            chunks.append({
                "id":      f"{paper['id'].split('/')[-1]}_0",
                "text":    text,
                "title":   paper["title"],
                "authors": paper["authors"],
                "year":    paper["year"],
                "url":     paper["url"],
                "query":   paper["query"],
            })
        else:
            start = 0
            chunk_idx = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "id":      f"{paper['id'].split('/')[-1]}_{chunk_idx}",
                    "text":    chunk_text,
                    "title":   paper["title"],
                    "authors": paper["authors"],
                    "year":    paper["year"],
                    "url":     paper["url"],
                    "query":   paper["query"],
                })
                start += chunk_size - overlap
                chunk_idx += 1

    logger.success(f"Created {len(chunks)} chunks from {len(papers)} papers")
    return chunks


def build_knowledge_base(config: dict, force_rebuild: bool = False):
    """Build or load ChromaDB knowledge base."""
    cfg = config["knowledge_base"]
    vs_cfg = config["vector_store"]

    persist_dir = Path(vs_cfg["persist_dir"])
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Check if collection already exists
    existing = [c.name for c in client.list_collections()]
    if vs_cfg["collection"] in existing and not force_rebuild:
        collection = client.get_collection(vs_cfg["collection"])
        count = collection.count()
        if count > 0:
            logger.info(f"Knowledge base exists: {count} chunks — skipping rebuild")
            return collection

    # Fetch papers
    papers = fetch_arxiv_papers(cfg["queries"], cfg["max_papers"])

    # Chunk
    chunks = chunk_papers(papers, cfg["chunk_size"], cfg["chunk_overlap"])

    # Embed
    logger.info(f"Loading embedding model: {vs_cfg['embedding_model']}")
    embedder = SentenceTransformer(vs_cfg["embedding_model"])

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, batch_size=64,
                                 show_progress_bar=True).tolist()

    # Index
    if vs_cfg["collection"] in existing:
        client.delete_collection(vs_cfg["collection"])

    collection = client.create_collection(
        name=vs_cfg["collection"],
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i:i+batch_size],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "title":   c["title"],
                "authors": c["authors"],
                "year":    str(c["year"]),
                "url":     c["url"],
                "query":   c["query"],
            } for c in batch]
        )

    logger.success(
        f"Knowledge base built: {collection.count()} chunks "
        f"from {len(papers)} papers"
    )
    return collection


def retrieve(collection, query: str, top_k: int = 5,
             embedder=None) -> list:
    """Retrieve top-k relevant chunks for a query."""
    if embedder is None:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "title":    results["metadatas"][0][i]["title"],
            "authors":  results["metadatas"][0][i]["authors"],
            "year":     results["metadatas"][0][i]["year"],
            "url":      results["metadatas"][0][i]["url"],
            "score":    1 - results["distances"][0][i],
        })
    return chunks


if __name__ == "__main__":
    cfg = load_config()
    collection = build_knowledge_base(cfg, force_rebuild=True)
    logger.info(f"Total chunks: {collection.count()}")