# CyberGuard — Multi-Agent Cybersecurity Research Assistant

A production-grade 5-agent LangGraph system for cybersecurity research — combining RAG over 818 indexed research papers, live threat intelligence, malware pattern analysis, and AI-powered quality evaluation.

Live Demo → [huggingface.co/spaces/dkamissah/cyberguard-agent](https://huggingface.co/spaces/dkamissah/cyberguard-agent)

## Agent Architecture

```
User Query → 🎯 Supervisor → 📚 RAG Agent → 🌐 Web Search
                                           → 🔬 Code Analyst
                                           → ✍️ Synthesiser → ⚖️ Critic
```

* **Supervisor** — routes query, orchestrates pipeline
* **RAG Agent** — retrieves from ChromaDB (818 cybersecurity papers)
* **Web Search** — live threat intelligence via Tavily API
* **Code Analyst** — MITRE ATT&CK mapping, malware pattern analysis
* **Synthesiser** — combines all findings into structured response
* **Critic** — scores quality (0–1), retries if below 0.7 (max 3x)

## Stack

* **Agent Framework:** LangGraph, LangChain
* **Vector Store:** ChromaDB + Sentence-Transformers
* **LLM:** LLaMA 3.1-8B via Groq API (~10s end-to-end)
* **Web Search:** Tavily API
* **API:** FastAPI
* **Dashboard:** Streamlit
* **CI/CD:** GitHub Actions

## Environment Variables

| Variable           | Where to get                                                                 |
| ------------------ | ---------------------------------------------------------------------------- |
| `GROQ_API_KEY`   | [console.groq.com](https://console.groq.com)— free                             |
| `HF_TOKEN`       | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)— free |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com)— free                                         |

## Running Locally

bash

```bash
git clone https://github.com/danielmissah/cyberguard-agent.git
cd cyberguard-agent
pip install -r requirements.txt

exportGROQ_API_KEY="gsk_..."
exportHF_TOKEN="hf_..."
exportTAVILY_API_KEY="tvly_..."

make kb         # build knowledge base
make api        # FastAPI at localhost:8001
make dashboard  # Streamlit at localhost:8501
```
