# CyberGuard — Multi-Agent Cybersecurity Research Assistant

A production-grade 5-agent LangGraph system for cybersecurity research — combining RAG over 818 indexed research papers, live threat intelligence, malware pattern analysis, and AI-powered quality evaluation.

**Live Demo → [cyberguard-agent.streamlit.app](https://cyberguard-agent.streamlit.app/)**

---

## Agent Architecture

```
User Query
    │
    ▼
🎯 Supervisor Agent        — routes query, orchestrates pipeline
    │
    ├──► 📚 RAG Agent       — retrieves from ChromaDB (818 cybersecurity papers)
    │
    ├──► 🌐 Web Search Agent — live threat intelligence via Tavily API
    │
    ├──► 🔬 Code Analyst     — MITRE ATT&CK mapping, malware pattern analysis
    │
    ├──► ✍️ Synthesiser      — combines all findings into structured response
    │
    └──► ⚖️ Critic Agent     — scores quality (0–1), retries if < 0.7 (max 3x)
```

---

## Key Features

| Feature                   | Details                                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| **Framework**       | LangGraph `StateGraph`with typed state and conditional routing             |
| **Knowledge Base**  | 818 arXiv cybersecurity papers, ChromaDB vector store                        |
| **Live Search**     | Tavily API for current CVEs and threat intelligence                          |
| **Quality Control** | Critic agent with retry loop — ensures responses meet 0.7 quality threshold |
| **LLM**             | LLaMA 3.1-8B via Groq API (~500 tokens/sec, ~10s end-to-end)                 |
| **Tracking**        | MLflow logs quality scores, latency, and source counts per query             |
| **API**             | FastAPI with 6 endpoints including graph introspection and batch queries     |
| **CI/CD**           | GitHub Actions — lint, test, Docker build, push to GHCR                     |

---

## Stack

* **Agent Framework:** LangGraph, LangChain
* **Vector Store:** ChromaDB + Sentence-Transformers (all-MiniLM-L6-v2)
* **LLM:** LLaMA 3.1-8B-Instant via Groq API
* **Web Search:** Tavily API
* **API:** FastAPI, Uvicorn, Pydantic v2
* **Tracking:** MLflow
* **Dashboard:** Streamlit, Plotly
* **Infrastructure:** Docker, GitHub Actions CI/CD
* **Language:** Python 3.11

---

## Running Locally

### 1. Clone and install

```bash
git clone https://github.com/danielamissah/cyberguard-agent.git
cd cyberguard-agent
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or export directly:

```bash
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"      # console.groq.com — free
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"            # huggingface.co — free
export TAVILY_API_KEY="tvly_xxxxxxxxxxxxxxxxxxxx"    # tavily.com — free (1000/month)
```

### 3. Build knowledge base

```bash
make kb   # fetches 818 cybersecurity papers from arXiv, indexes into ChromaDB
```

### 4. Start API and dashboard

```bash
make api        # FastAPI at http://localhost:8001
make dashboard  # Streamlit at http://localhost:8501
```

---

## Environment Variables

| Variable                | Required    | Where to get                                                                 |
| ----------------------- | ----------- | ---------------------------------------------------------------------------- |
| `GROQ_API_KEY`        | ✅ Yes      | [console.groq.com](https://console.groq.com/)— free                            |
| `HF_TOKEN`            | ✅ Yes      | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)— free |
| `TAVILY_API_KEY`      | ✅ Yes      | [tavily.com](https://tavily.com/)— free (1,000 searches/month)                 |
| `MLFLOW_TRACKING_URI` | ❌ Optional | Leave blank to skip experiment tracking                                      |

---

## API Endpoints

| Method | Endpoint                    | Description                             |
| ------ | --------------------------- | --------------------------------------- |
| GET    | `/health`                 | Health check                            |
| GET    | `/stats`                  | Knowledge base and agent stats          |
| POST   | `/query`                  | Run full 5-agent pipeline               |
| POST   | `/query/batch`            | Batch queries (max 5)                   |
| GET    | `/graph/info`             | Agent graph structure and routing logic |
| POST   | `/knowledge-base/rebuild` | Rebuild ChromaDB index from arXiv       |

---

## Example

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are adversarial attacks against ML-based intrusion detection systems?", "track": false}'
```

Response includes structured findings, quality score (0–1), RAG paper sources, live web sources, and full agent execution trace.

---

## Project Structure

```
cyberguard-agent/
├── src/
│   ├── graph/
│   │   └── agent_graph.py     # LangGraph StateGraph — all 5 agents
│   ├── tools/
│   │   └── knowledge_base.py  # arXiv fetcher, chunker, ChromaDB indexer
│   └── api/
│       └── main.py            # FastAPI application
├── dashboard/
│   └── app.py                 # Streamlit dashboard (3 tabs)
├── configs/
│   └── config.yaml            # LLM, agent, and vector store config
├── tests/
│   └── test_smoke.py          # Smoke tests (pytest)
├── .github/workflows/
│   └── ci_cd.yml              # Lint → test → Docker build → push to GHCR
├── .env.example               # Environment variable template
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Knowledge Base Topics

Papers indexed from arXiv across 10 cybersecurity domains:

* Adversarial ML attacks on cybersecurity systems
* Ransomware detection and classification
* Network intrusion detection with deep learning
* Data poisoning attacks on neural networks
* Federated learning security and privacy
* Malware classification with neural networks
* Phishing detection using NLP
* Anomaly detection for cybersecurity
* Cyber threat intelligence with ML
* LLM security vulnerabilities and prompt injection


---
## title: CyberGuard — Multi-Agent Cybersecurity Research Assistant
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: dashboard/app.py
pinned: false
license: mit
short_description: 5-agent LangGraph system for cybersecurity research — RAG + live threat intel


# CyberGuard — Multi-Agent Cybersecurity Research Assistant


A production-grade 5-agent LangGraph system for cybersecurity research — combining RAG over 818 indexed research papers, live threat intelligence, malware pattern analysis, and AI-powered quality evaluation.


## Agent Architecture


```
User Query
    │
    ▼
🎯 Supervisor Agent        — routes query, orchestrates pipeline
    │
    ├──► 📚 RAG Agent       — retrieves from ChromaDB (818 cybersecurity papers)
    ├──► 🌐 Web Search Agent — live threat intelligence via Tavily API
    ├──► 🔬 Code Analyst     — MITRE ATT&CK mapping, malware pattern analysis
    ├──► ✍️ Synthesiser      — combines all findings into structured response
    └──► ⚖️ Critic Agent     — scores quality (0–1), retries if < 0.7 (max 3x)
```


## Stack


* **Agent Framework:** LangGraph, LangChain
* **Vector Store:** ChromaDB + Sentence-Transformers
* **LLM:** LLaMA 3.1-8B via Groq API
* **Web Search:** Tavily API
* **API:** FastAPI
* **Dashboard:** Streamlit
* **CI/CD:** GitHub Actions → Docker → HF Spaces


## Environment Variables Required


| Variable             | Where to get                                                                 |
| ---------------------- | ------------------------------------------------------------------------------ |
| `GROQ_API_KEY`   | [console.groq.com](https://console.groq.com/)— free                            |
| `HF_TOKEN`       | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)— free |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com/)— free                                        |


## Running Locally


```bash
git clone https://github.com/dkamissah/cyberguard-agent.git
cd cyberguard-agent
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."
export HF_TOKEN="hf_..."
export TAVILY_API_KEY="tvly_..."

make kb         # build knowledge base (~5 min)
make api        # FastAPI at localhost:8001
make dashboard  # Streamlit at localhost:8501
```


## Author


**Daniel Kwame Amissah** — ML Engineer · Hamburg, Germany
[LinkedIn](https://linkedin.com/in/danielkamissah) · [GitHub](https://github.com/dkamissah)
---
