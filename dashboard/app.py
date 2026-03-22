"""
CyberGuard — Multi-Agent Research Assistant Dashboard
======================================================
Interactive Streamlit UI for the 5-agent LangGraph system.
"""

import os
import json
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="CyberGuard — AI Research Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("API_URL", "http://localhost:8001")
HF_SPACE = os.environ.get("SPACE_ID", "") != ""  # True when running on HF Spaces

AGENT_COLORS = {
    "supervisor":    "#3498DB",
    "rag":           "#2ECC71",
    "web_search":    "#E67E22",
    "code_analysis": "#9B59B6",
    "synthesiser":   "#1ABC9C",
    "critic":        "#E74C3C",
}

SAMPLE_QUERIES = [
    "What are the latest adversarial attack techniques against ML-based intrusion detection systems?",
    "How do ransomware groups use living-off-the-land techniques to evade detection?",
    "What are the most effective defences against prompt injection attacks on LLMs?",
    "Explain data poisoning attacks on federated learning systems and countermeasures.",
    "What MITRE ATT&CK techniques are commonly used in APT campaigns targeting critical infrastructure?",
    "How can anomaly detection models be hardened against adversarial evasion attacks?",
]

def api_get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        return r.json()
    except requests.Timeout:
        return {"error": f"Request timed out after {timeout}s — try again"}
    except Exception as e:
        return {"error": str(e)}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ CyberGuard")
    st.caption("Multi-Agent Cybersecurity Research Assistant")
    st.divider()

    stats = api_get("/stats")
    if "error" not in stats:
        st.metric("Knowledge Base", f"{stats.get('knowledge_base_chunks', 0):,} chunks")
        st.metric("Active Agents", stats.get("agents", 5))
        st.metric("LLM Model", "Mistral-7B")
    else:
        st.warning("API offline — start with `make api`")

    st.divider()
    st.caption("**Agent Pipeline**")
    agents = ["🎯 Supervisor", "📚 RAG", "🌐 Web Search",
              "🔬 Code Analysis", "✍️ Synthesiser", "⚖️ Critic"]
    for agent in agents:
        st.caption(f"  {agent}")

    st.divider()
    st.caption("**Stack:** LangGraph · LangChain · ChromaDB · Mistral-7B · Tavily · FastAPI · MLflow")
    st.caption("**GitHub:** [cyberguard-agent](https://github.com/dkamissah/cyberguard-agent)")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Research Assistant", "Agent Trace", "System Info"])


# ── TAB 1: Research Assistant ─────────────────────────────────────────────────
with tab1:
    st.title("🛡️ CyberGuard Research Assistant")
    st.caption("5-agent LangGraph system — RAG + Live Threat Intelligence + Code Analysis")

    # Session state for result and query
    if "cg_result" not in st.session_state:
        st.session_state.cg_result = None
    if "cg_query" not in st.session_state:
        st.session_state.cg_query = ""

    # ── Show result or input form ─────────────────────────────────────────────
    if st.session_state.cg_result is not None:
        result = st.session_state.cg_result

        # Process new query button
        if st.button("🔄 Process New Query", type="primary"):
            st.session_state.cg_result = None
            st.session_state.cg_query = ""
            st.rerun()

        st.divider()

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quality Score",  f"{result['quality_score']:.0%}")
        m2.metric("RAG Sources",    result["n_rag_results"])
        m3.metric("Web Sources",    result["n_web_results"])
        m4.metric("Latency",        f"{result['latency_ms']:.0f}ms")

        st.divider()
        st.subheader("Research Findings")
        st.markdown(result["response"])
        st.divider()

        # Sources
        if result.get("sources"):
            st.subheader("Sources")
            papers = [s for s in result["sources"] if s["type"] == "paper"]
            web    = [s for s in result["sources"] if s["type"] == "web"]
            if papers:
                st.caption("**Research Papers**")
                for s in papers[:5]:
                    st.markdown(f"- [{s['title']}]({s['url']}) — relevance: {s.get('score', 0):.3f}")
            if web:
                st.caption("**Live Intelligence**")
                for s in web[:5]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

        # Agent trace
        if result.get("agent_trace"):
            with st.expander("Agent execution trace"):
                for step in result["agent_trace"]:
                    agent = step.get("agent", "unknown")
                    color = AGENT_COLORS.get(agent, "#888")
                    st.markdown(
                        f'<span style="color:{color}">●</span> **{agent.title()}** — '
                        + ", ".join(f"{k}: {v}" for k, v in step.items() if k != "agent"),
                        unsafe_allow_html=True
                    )

    else:
        # ── Input form ────────────────────────────────────────────────────────
        query = st.text_area(
            "Enter your cybersecurity research question:",
            value=st.session_state.cg_query,
            height=100,
            placeholder="e.g. What are the latest adversarial attack techniques against ML-based IDS?"
        )

        st.caption("**Sample queries — click to load:**")
        for i, q in enumerate(SAMPLE_QUERIES):
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.cg_query = q
                st.rerun()

        st.divider()
        run = st.button("🔍 Analyse", type="primary", use_container_width=True)

        if run and query.strip():
            # ── Agent progress bars ───────────────────────────────────────────
            st.markdown("**Running 5-agent pipeline...**")

            agents_pipeline = [
                ("🎯 Supervisor",     "supervisor",    "Routing query to agents"),
                ("📚 RAG Agent",      "rag",           "Retrieving research papers"),
                ("🌐 Web Search",     "web_search",    "Searching threat intelligence"),
                ("🔬 Code Analyst",   "code_analysis", "Analysing malware patterns"),
                ("⚖️ Critic",         "critic",        "Evaluating response quality"),
            ]

            progress_bars = []
            status_texts  = []

            for name, key, desc in agents_pipeline:
                color = AGENT_COLORS.get(key, "#888")
                st.markdown(
                    f'<span style="color:{color}">●</span> **{name}** — {desc}',
                    unsafe_allow_html=True
                )
                pb = st.progress(0)
                st.caption(f"  ⏳ Waiting...")
                progress_bars.append(pb)
                status_texts.append(st.empty())

            # Animate progress while waiting for API
            import threading, time as _time

            result_holder = {}
            done_flag = threading.Event()

            def fetch_result():
                result_holder["data"] = api_post(
                    "/query", {"query": query, "track": False}, timeout=300
                )
                done_flag.set()

            thread = threading.Thread(target=fetch_result)
            thread.start()

            # Simulate per-agent progress
            agent_times = [0.5, 2.0, 3.5, 5.0, 6.5]
            completed   = [False] * 5

            while not done_flag.is_set():
                elapsed = _time.time()
                for i, t in enumerate(agent_times):
                    if not completed[i]:
                        frac = min(1.0, (elapsed % 10) / t) if t > 0 else 1.0
                        progress_bars[i].progress(min(int(frac * 80), 80))
                _time.sleep(0.3)

            # Mark all complete
            for i, pb in enumerate(progress_bars):
                pb.progress(100)

            thread.join()
            result = result_holder.get("data", {})

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.session_state.cg_result = result
                st.session_state.cg_query  = query
                st.rerun()

        elif run:
            st.warning("Please enter a query first.")


# ── TAB 2: Agent Trace Visualisation ─────────────────────────────────────────
with tab2:
    st.subheader("Agent Pipeline Visualisation")
    st.caption("How the 5 agents collaborate to answer your query")

    # Pipeline diagram
    fig = go.Figure()

    nodes = [
        ("Supervisor", 0.1, 0.5, "Routes query\nto agents"),
        ("RAG Agent",  0.3, 0.8, "Retrieves\nresearch papers"),
        ("Web Search", 0.3, 0.2, "Live threat\nintelligence"),
        ("Code Analyst", 0.55, 0.5, "Malware &\nexploit analysis"),
        ("Synthesiser", 0.75, 0.5, "Combines all\nfindings"),
        ("Critic", 0.92, 0.5, "Quality\nevaluation"),
    ]

    colors = ["#3498DB", "#2ECC71", "#E67E22", "#9B59B6", "#1ABC9C", "#E74C3C"]

    for (name, x, y, desc), color in zip(nodes, colors):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=50, color=color, symbol="circle"),
            text=[name],
            textposition="top center",
            hovertext=desc,
            hoverinfo="text",
            showlegend=False,
        ))

    # Edges
    edges = [
        (0.1, 0.5, 0.3, 0.8),
        (0.1, 0.5, 0.3, 0.2),
        (0.3, 0.8, 0.55, 0.5),
        (0.3, 0.2, 0.55, 0.5),
        (0.55, 0.5, 0.75, 0.5),
        (0.75, 0.5, 0.92, 0.5),
        (0.92, 0.5, 0.75, 0.5),  # retry loop
    ]

    for x0, y0, x1, y1 in edges:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="#666", arrowwidth=2,
        )

    fig.update_layout(
        height=400, template="plotly_dark",
        xaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Agent descriptions
    st.subheader("Agent Roles")
    cols = st.columns(5)
    agent_info = [
        ("🎯 Supervisor",     "#3498DB", "Routes the query and decides which agents to activate based on query type"),
        ("📚 RAG Agent",      "#2ECC71", "Retrieves relevant chunks from 200+ indexed cybersecurity research papers"),
        ("🌐 Web Search",     "#E67E22", "Searches live threat intelligence databases and CVE feeds via Tavily"),
        ("🔬 Code Analyst",   "#9B59B6", "Analyses malware patterns and maps to MITRE ATT&CK framework"),
        ("⚖️ Critic",         "#E74C3C", "Scores response quality 0-1 and triggers retry loop if below 0.7"),
    ]
    for col, (name, color, desc) in zip(cols, agent_info):
        col.markdown(
            f'<div style="border-left: 4px solid {color}; padding-left: 8px;">'
            f'<b>{name}</b><br><small>{desc}</small></div>',
            unsafe_allow_html=True
        )


# ── TAB 3: System Info ────────────────────────────────────────────────────────
with tab3:
    st.subheader("System Information")

    info = api_get("/graph/info")
    stats = api_get("/stats")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Agent Pipeline")
        if "error" not in info:
            for edge in info.get("edges", []):
                st.code(edge)
            st.subheader("Node Details")
            for node in info.get("nodes", []):
                st.markdown(f"**{node['name']}** — {node['role']}")

    with col2:
        st.subheader("Knowledge Base Stats")
        if "error" not in stats:
            st.metric("Total Chunks",    f"{stats.get('knowledge_base_chunks', 0):,}")
            st.metric("Active Agents",   stats.get("agents", 5))
            st.metric("Embedding Model", "all-MiniLM-L6-v2")
            st.metric("LLM",             "Mistral-7B-Instruct-v0.3")
            st.metric("Vector Store",    "ChromaDB")
            st.metric("Framework",       "LangGraph + LangChain")

        st.divider()
        if st.button("Rebuild Knowledge Base"):
            with st.spinner("Rebuilding..."):
                result = api_post("/knowledge-base/rebuild", {})
            st.success("Knowledge base rebuilt")