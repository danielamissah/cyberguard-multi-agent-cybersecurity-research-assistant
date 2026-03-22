"""Smoke tests for CyberGuard Agent"""
import pytest


def test_config_loads():
    from src.graph.agent_graph import load_config
    cfg = load_config()
    assert "llm" in cfg
    assert "agents" in cfg
    assert "vector_store" in cfg


def test_agent_state_structure():
    from src.graph.agent_graph import AgentState
    state: AgentState = {
        "query": "test",
        "messages": [],
        "rag_results": [],
        "web_results": [],
        "code_analysis": "",
        "draft_response": "",
        "final_response": "",
        "quality_score": 0.0,
        "iteration": 0,
        "next_agent": "rag",
        "sources": [],
        "agent_trace": [],
    }
    assert state["query"] == "test"
    assert state["quality_score"] == 0.0


def test_graph_builds():
    from src.graph.agent_graph import build_graph, load_config
    cfg   = load_config()
    graph = build_graph(cfg)
    assert graph is not None


def test_knowledge_base_config():
    from src.graph.agent_graph import load_config
    cfg = load_config()
    assert len(cfg["knowledge_base"]["queries"]) >= 5
    assert cfg["knowledge_base"]["max_papers"] > 0


def test_agent_colors_complete():
    agents = ["supervisor", "rag", "web_search", "code_analysis", "synthesiser", "critic"]
    assert len(agents) == 6
