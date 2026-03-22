
# CyberGuard — Multi-Agent Cybersecurity Research Assistant

**Live Demo → [huggingface.co/spaces/dkamissah/cyberguard-agent](https://huggingface.co/spaces/dkamissah/cyberguard-agent)**

A production-grade 5-agent LangGraph system for cybersecurity research — combining RAG over 818 indexed arXiv papers, live threat intelligence, malware pattern analysis, and AI-powered quality evaluation. End-to-end latency ~7.6 seconds.

---

## Key Results

| Metric                     | Value                              |
| -------------------------- | ---------------------------------- |
| Knowledge Base             | 818 arXiv cybersecurity papers     |
| Quality Score              | 85% (Critic agent evaluation)      |
| RAG Sources per query      | 5 relevant papers                  |
| Live Web Sources per query | 5 threat intelligence sources      |
| End-to-end latency         | ~7.6 seconds                       |
| LLM                        | LLaMA 3.1-8B via Groq (~500 tok/s) |

---

## Agent Architecture

```
User Query
    │
    ▼
🎯 Supervisor Agent     — routes query, orchestrates pipeline
    │
    ├──► 📚 RAG Agent       — ChromaDB retrieval (818 cybersecurity papers)
    ├──► 🌐 Web Search       — live threat intelligence via Tavily API
    ├──► 🔬 Code Analyst     — MITRE ATT&CK mapping, malware pattern analysis
    ├──► ✍️ Synthesiser      — structured response with executive summary
    └──► ⚖️ Critic Agent     — quality score 0–1, retries if below 0.7 (max 3x)
```

### LangGraph State Machine

* Typed `AgentState` with full conversation history
* Conditional routing — Critic triggers retry loop if quality < 0.7
* Max 3 iterations before fallback to best available response
* All agent outputs logged to MLflow per query

---

## Stack

* **Agent Framework:** LangGraph, LangChain
* **Vector Store:** ChromaDB + Sentence-Transformers (all-MiniLM-L6-v2)
* **LLM:** LLaMA 3.1-8B-Instant via Groq API
* **Web Search:** Tavily API
* **API:** FastAPI (9 endpoints)
* **Dashboard:** Streamlit (3 tabs — Research Assistant, Agent Trace, System Info)
* **Tracking:** MLflow
* **Infrastructure:** Docker, GitHub Actions CI/CD → HF Spaces

---

## Dashboard

| Tab                          | Contents                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Research Assistant** | Query input, sample queries, per-agent progress bars, structured results with sources |
| **Agent Trace**        | Interactive pipeline diagram, agent role descriptions                                 |
| **System Info**        | Knowledge base stats, graph structure, live API test                                  |

---

## Running Locally

bash

```bash
git clone https://github.com/danielamissah/cyberguard-agent.git
cd cyberguard-agent
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

make kb         # build knowledge base (~5 min)
make api        # FastAPI at localhost:8001
make dashboard  # Streamlit at localhost:8501
```

## Environment Variables

| Variable                | Required | Where to get                                                                 |
| ----------------------- | -------- | ---------------------------------------------------------------------------- |
| `GROQ_API_KEY`        | ✅       | [console.groq.com](https://console.groq.com)— free                             |
| `HF_TOKEN`            | ✅       | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)— free |
| `TAVILY_API_KEY`      | ✅       | [tavily.com](https://tavily.com)— free (1,000/month)                           |
| `MLFLOW_TRACKING_URI` | ❌       | Optional — skip to disable tracking                                         |

---

## API Endpoints

| Method | Endpoint                    | Description                    |
| ------ | --------------------------- | ------------------------------ |
| GET    | `/health`                 | Health check                   |
| GET    | `/stats`                  | Knowledge base and agent stats |
| POST   | `/query`                  | Run full 5-agent pipeline      |
| POST   | `/query/batch`            | Batch queries (max 5)          |
| GET    | `/graph/info`             | Agent graph structure          |
| POST   | `/knowledge-base/rebuild` | Rebuild ChromaDB from arXiv    |

---

## Knowledge Base Topics

818 papers indexed across 10 cybersecurity domains:
adversarial ML · ransomware detection · network intrusion detection ·
data poisoning · federated learning security · malware classification ·
phishing detection · anomaly detection · cyber threat intelligence · LLM security

---

## Project Structure

```
cyberguard-agent/
├── src/
│   ├── graph/agent_graph.py    # LangGraph StateGraph — all 5 agents
│   ├── tools/knowledge_base.py # arXiv fetcher, chunker, ChromaDB indexer
│   └── api/main.py             # FastAPI application
├── dashboard/app.py            # Streamlit dashboard (3 tabs)
├── configs/config.yaml         # All hyperparameters
├── tests/test_smoke.py         # Smoke tests (pytest)
├── .github/workflows/ci_cd.yml # Lint → test → Docker → GHCR
├── .env.example                # Environment variable template
├── Dockerfile
├── Makefile
└── requirements.txt
```
