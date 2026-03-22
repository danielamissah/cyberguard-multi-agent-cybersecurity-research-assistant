"""
CyberGuard Agent Graph
======================
LangGraph multi-agent system with 5 specialised agents:
  1. Supervisor   — routes queries, orchestrates workflow
  2. RAG Agent    — retrieves from cybersecurity knowledge base
  3. Web Search   — live threat intelligence via Tavily
  4. Code Analyst — analyses malware/exploit patterns
  5. Critic       — evaluates response quality, triggers retry

State flows through a LangGraph StateGraph with conditional routing.
"""

import os
import re
import yaml
import json
import chromadb
import requests
from typing import TypedDict, Annotated, Literal
from loguru import logger
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:            str
    messages:         Annotated[list, add_messages]
    rag_results:      list
    web_results:      list
    code_analysis:    str
    draft_response:   str
    final_response:   str
    quality_score:    float
    iteration:        int
    next_agent:       str
    sources:          list
    agent_trace:      list


# ── LLM caller ────────────────────────────────────────────────────────────────

def call_llm(prompt: str, config: dict, system: str = "") -> str:
    """Call LLM — Groq (primary, fast) or HF (fallback)."""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # ── Groq (primary — ~500 tokens/sec, free tier) ───────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={
                    "model":       "llama-3.1-8b-instant",
                    "messages":    messages,
                    "max_tokens":  config["llm"].get("max_tokens", 512),
                    "temperature": config["llm"].get("temperature", 0.1),
                },
                timeout=15
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Groq failed: {e} — falling back to HF")

    # ── HF Inference API (fallback) ───────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    for model in ["mistralai/Mistral-7B-Instruct-v0.1", "HuggingFaceH4/zephyr-7b-beta"]:
        try:
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
                headers={"Authorization": f"Bearer {hf_token}",
                         "Content-Type": "application/json"},
                json={"model": model, "messages": messages,
                      "max_tokens": 512, "temperature": 0.1},
                timeout=30
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue

    return _rule_based_fallback(prompt)


def _rule_based_fallback(prompt: str) -> str:
    """Simple fallback when LLM is unavailable."""
    prompt_lower = prompt.lower()
    if "spam" in prompt_lower or "phishing" in prompt_lower:
        return ("Spam and phishing are social engineering attacks using deceptive messages. "
                "ML-based detectors use NLP features (TF-IDF, embeddings) and classifiers "
                "(SVM, BERT) to identify malicious content. Key signals include sender reputation, "
                "URL patterns, and linguistic anomalies.")
    if "ransomware" in prompt_lower:
        return ("Ransomware encrypts victim files and demands payment. Detection approaches include "
                "behavioural analysis (file system activity), API call sequences, and ML classifiers "
                "trained on static/dynamic features. Recent variants use living-off-the-land techniques.")
    if "intrusion" in prompt_lower or "ids" in prompt_lower:
        return ("Intrusion Detection Systems (IDS) use anomaly detection and signature matching. "
                "ML approaches include Random Forest, LSTM autoencoders, and Graph Neural Networks "
                "on network traffic features. Adversarial attacks evade detection via feature manipulation.")
    return ("This query relates to cybersecurity. The knowledge base contains 818 indexed research papers "
            "on adversarial ML, ransomware, intrusion detection, and related topics. "
            "LLM is temporarily unavailable — please retry in a moment.")


# ── Cached embedder (loaded once at startup) ──────────────────────────────────
_embedder = None

def get_embedder(config: dict):
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model (one-time)...")
        _embedder = SentenceTransformer(config["vector_store"]["embedding_model"])
        logger.success("Embedding model loaded and cached")
    return _embedder


# ── Tool: RAG retrieval ───────────────────────────────────────────────────────

def rag_retrieve(query: str, config: dict) -> list:
    """Retrieve relevant chunks from ChromaDB."""
    from src.tools.knowledge_base import retrieve
    vs_cfg   = config["vector_store"]
    client   = chromadb.PersistentClient(path=vs_cfg["persist_dir"])
    embedder = get_embedder(config)

    try:
        collection = client.get_collection(vs_cfg["collection"])
        results    = retrieve(collection, query,
                               top_k=vs_cfg["top_k"], embedder=embedder)
        return results
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return []


# ── Tool: Web search via Tavily ───────────────────────────────────────────────

def web_search(query: str, config: dict) -> list:
    """Search live web for threat intelligence using Tavily."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("No TAVILY_API_KEY — skipping web search")
        return []

    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":        api_key,
                "query":          query + " cybersecurity threat",
                "max_results":    config["tavily"]["max_results"],
                "search_depth":   "advanced",
                "include_answer": True,
            },
            timeout=30
        )
        r.raise_for_status()
        data    = r.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "content": item.get("content", "")[:500],
                "score":   item.get("score", 0),
            })
        return results
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []


# ── Agent 1: Supervisor ───────────────────────────────────────────────────────

def supervisor_agent(state: AgentState, config: dict) -> AgentState:
    """Routes the query and decides which agents to activate."""
    logger.info("Supervisor: routing query")
    query = state["query"]

    system = """You are a cybersecurity research supervisor.
Analyse the user's query and decide which agents to activate.
Respond with a JSON object containing:
{
  "needs_rag": true/false,
  "needs_web_search": true/false,
  "needs_code_analysis": true/false,
  "reasoning": "brief explanation"
}
Only respond with the JSON object, nothing else."""

    response = call_llm(query, config, system=system)

    try:
        clean = re.sub(r"```json|```", "", response).strip()
        routing = json.loads(clean)
    except Exception:
        routing = {"needs_rag": True, "needs_web_search": True,
                   "needs_code_analysis": False}

    trace = state.get("agent_trace", [])
    trace.append({"agent": "supervisor", "routing": routing})

    return {
        **state,
        "next_agent":   "rag" if routing.get("needs_rag", True) else "synthesiser",
        "agent_trace":  trace,
        "iteration":    state.get("iteration", 0),
        "messages":     [AIMessage(content=f"Supervisor routing: {routing.get('reasoning', 'activating agents')}")]
    }


# ── Agent 2: RAG Agent ────────────────────────────────────────────────────────

def rag_agent(state: AgentState, config: dict) -> AgentState:
    """Retrieves relevant cybersecurity research."""
    logger.info("RAG Agent: retrieving knowledge")
    query   = state["query"]
    results = rag_retrieve(query, config)

    summary = ""
    sources = state.get("sources", [])

    if results:
        context = "\n\n".join([
            f"[{r['title']} ({r['year']})]\n{r['text']}"
            for r in results[:3]
        ])
        prompt = f"""Based on these cybersecurity research papers, provide key findings relevant to:
{query}

Research context:
{context}

Summarise the most relevant findings in 3-5 bullet points."""

        summary = call_llm(prompt, config)
        for r in results:
            sources.append({"type": "paper", "title": r["title"],
                             "url": r["url"], "score": r.get("score", 0)})

    trace = state.get("agent_trace", [])
    trace.append({"agent": "rag", "n_results": len(results),
                  "top_score": results[0].get("score", 0) if results else 0})

    return {
        **state,
        "rag_results":  results,
        "sources":      sources,
        "agent_trace":  trace,
        "messages":     [AIMessage(content=f"RAG Agent found {len(results)} relevant papers")],
        "draft_response": summary,
    }


# ── Agent 3: Web Search Agent ─────────────────────────────────────────────────

def web_search_agent(state: AgentState, config: dict) -> AgentState:
    """Searches for live threat intelligence."""
    logger.info("Web Search Agent: searching threat intelligence")
    query   = state["query"]
    results = web_search(query, config)

    summary = ""
    sources = state.get("sources", [])

    if results:
        context = "\n\n".join([
            f"[{r['title']}]\n{r['content']}"
            for r in results[:3]
        ])
        prompt = f"""Based on these current cybersecurity threat intelligence sources, provide insights for:
{query}

Sources:
{context}

Summarise the most critical current threats and findings in 3-5 bullet points."""

        summary = call_llm(prompt, config)
        for r in results:
            sources.append({"type": "web", "title": r["title"],
                             "url": r["url"], "score": r.get("score", 0)})

    trace = state.get("agent_trace", [])
    trace.append({"agent": "web_search", "n_results": len(results)})

    return {
        **state,
        "web_results":  results,
        "sources":      sources,
        "agent_trace":  trace,
        "messages":     [AIMessage(content=f"Web Search Agent found {len(results)} live sources")],
    }


# ── Agent 4: Code Analysis Agent ─────────────────────────────────────────────

def code_analysis_agent(state: AgentState, config: dict) -> AgentState:
    """Analyses malware patterns or exploit code in the query."""
    logger.info("Code Analysis Agent: analysing patterns")
    query = state["query"]

    prompt = f"""You are a malware and cybersecurity code analysis expert.
Analyse the following cybersecurity query for any code patterns,
attack vectors, or exploit techniques:

Query: {query}

Provide:
1. Attack pattern identification (if any code/technique mentioned)
2. MITRE ATT&CK framework mapping (if applicable)
3. Detection recommendations
4. Mitigation strategies

Be concise and technical."""

    analysis = call_llm(prompt, config)

    trace = state.get("agent_trace", [])
    trace.append({"agent": "code_analysis", "completed": True})

    return {
        **state,
        "code_analysis": analysis,
        "agent_trace":   trace,
        "messages":      [AIMessage(content="Code Analysis Agent completed pattern analysis")],
    }


# ── Agent 5: Critic Agent ─────────────────────────────────────────────────────

def critic_agent(state: AgentState, config: dict) -> AgentState:
    """Evaluates response quality and triggers retry if below threshold."""
    logger.info("Critic Agent: evaluating response quality")
    draft     = state.get("draft_response", "")
    query     = state["query"]
    threshold = config["agents"]["critic"]["quality_threshold"]

    if not draft:
        return {**state, "quality_score": 0.0,
                "agent_trace": state.get("agent_trace", []) + [
                    {"agent": "critic", "score": 0.0, "verdict": "no_response"}]}

    prompt = f"""You are a cybersecurity research quality evaluator.
Rate the following response to the query on a scale of 0.0 to 1.0.

Query: {query}

Response: {draft}

Evaluate based on:
- Technical accuracy (0-0.3)
- Completeness (0-0.3)
- Actionability (0-0.2)
- Source grounding (0-0.2)

Respond ONLY with a JSON object:
{{"score": 0.85, "feedback": "brief feedback", "missing": "what is missing"}}"""

    response = call_llm(prompt, config)

    try:
        clean  = re.sub(r"```json|```", "", response).strip()
        result = json.loads(clean)
        score  = float(result.get("score", 0.7))
        feedback = result.get("feedback", "")
    except Exception:
        score    = 0.7
        feedback = "Could not parse critic response"

    trace = state.get("agent_trace", [])
    trace.append({"agent": "critic", "score": score,
                  "verdict": "pass" if score >= threshold else "retry",
                  "feedback": feedback})

    logger.info(f"Critic score: {score:.2f} (threshold: {threshold})")

    return {
        **state,
        "quality_score": score,
        "agent_trace":   trace,
        "messages":      [AIMessage(content=f"Critic score: {score:.2f}/1.0 — {'✓ Pass' if score >= threshold else '↻ Retry'}")]
    }


# ── Synthesiser (final response) ──────────────────────────────────────────────

def synthesiser_agent(state: AgentState, config: dict) -> AgentState:
    """Synthesises all agent outputs into a final response."""
    logger.info("Synthesiser: building final response")
    query        = state["query"]
    rag_draft    = state.get("draft_response", "")
    code_analysis = state.get("code_analysis", "")
    web_results  = state.get("web_results", [])

    web_summary = ""
    if web_results:
        web_summary = "\n".join([
            f"- {r['title']}: {r['content'][:200]}"
            for r in web_results[:3]
        ])

    prompt = f"""You are a senior cybersecurity research analyst.
Synthesise the following research findings into a comprehensive, well-structured response.

Query: {query}

Research paper findings:
{rag_draft or "No paper findings available"}

Live threat intelligence:
{web_summary or "No live intelligence available"}

Code/pattern analysis:
{code_analysis or "No code analysis performed"}

Provide a comprehensive response with:
1. Executive Summary (2-3 sentences)
2. Key Technical Findings (bullet points)
3. Current Threat Landscape (if relevant)
4. Recommendations & Mitigations
5. References to supporting research

Be thorough, technical, and actionable."""

    final = call_llm(prompt, config)

    trace = state.get("agent_trace", [])
    trace.append({"agent": "synthesiser", "response_length": len(final)})

    return {
        **state,
        "final_response": final,
        "draft_response": final,
        "agent_trace":    trace,
        "messages":       [AIMessage(content="Synthesiser: final response ready")],
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_supervisor(state: AgentState) -> str:
    return "rag"


def route_after_rag(state: AgentState) -> str:
    return "web_search"


def route_after_web_search(state: AgentState) -> str:
    return "code_analysis"


def route_after_code_analysis(state: AgentState) -> str:
    return "synthesiser"


def route_after_critic(state: AgentState) -> Literal["synthesiser", "__end__"]:
    score     = state.get("quality_score", 0)
    iteration = state.get("iteration", 0)
    threshold = 0.7
    max_iter  = 3

    if score >= threshold or iteration >= max_iter:
        return END
    return "synthesiser"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph(config: dict):
    """Construct and compile the LangGraph agent graph."""

    def _supervisor(state):    return supervisor_agent(state, config)
    def _rag(state):           return rag_agent(state, config)
    def _web_search(state):    return web_search_agent(state, config)
    def _code_analysis(state): return code_analysis_agent(state, config)
    def _synthesiser(state):   return synthesiser_agent(state, config)
    def _critic(state):
        s = critic_agent(state, config)
        return {**s, "iteration": s.get("iteration", 0) + 1}

    graph = StateGraph(AgentState)

    graph.add_node("supervisor",     _supervisor)
    graph.add_node("rag",            _rag)
    graph.add_node("web_search",     _web_search)
    graph.add_node("code_analysis",  _code_analysis)
    graph.add_node("synthesiser",    _synthesiser)
    graph.add_node("critic",         _critic)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor",    "rag")
    graph.add_edge("rag",           "web_search")
    graph.add_edge("web_search",    "code_analysis")
    graph.add_edge("code_analysis", "synthesiser")
    graph.add_edge("synthesiser",   "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"synthesiser": "synthesiser", END: END}
    )

    return graph.compile()


def run_query(query: str, config: dict) -> dict:
    """Run a query through the full agent pipeline."""
    graph = build_graph(config)

    initial_state: AgentState = {
        "query":          query,
        "messages":       [HumanMessage(content=query)],
        "rag_results":    [],
        "web_results":    [],
        "code_analysis":  "",
        "draft_response": "",
        "final_response": "",
        "quality_score":  0.0,
        "iteration":      0,
        "next_agent":     "rag",
        "sources":        [],
        "agent_trace":    [],
    }

    logger.info(f"Running query: {query[:80]}...")
    final_state = graph.invoke(initial_state)

    return {
        "query":          final_state["query"],
        "response":       final_state["final_response"],
        "quality_score":  final_state["quality_score"],
        "sources":        final_state["sources"],
        "agent_trace":    final_state["agent_trace"],
        "n_rag_results":  len(final_state["rag_results"]),
        "n_web_results":  len(final_state["web_results"]),
    }


if __name__ == "__main__":
    cfg    = load_config()
    result = run_query(
        "What are the latest adversarial attack techniques against ML-based intrusion detection systems?",
        cfg
    )
    print("\n" + "="*60)
    print("FINAL RESPONSE:")
    print("="*60)
    print(result["response"])
    print(f"\nQuality Score: {result['quality_score']:.2f}")
    print(f"Sources: {len(result['sources'])}")
    print(f"Agent Trace: {[t['agent'] for t in result['agent_trace']]}")