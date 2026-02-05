# 🎬 YouTube Video Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/PostgreSQL-16-blue.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  A production-grade video recommendation system featuring two-tower architecture, multi-stage ranking, LLM-powered chatbot, and real-time monitoring.
</p>

---

## ✨ Features

- **Two-Tower Retrieval** — CLIP & Sentence Transformer embeddings with FAISS indexing
- **Multi-Stage Ranking** — Deep Cross Network + diversity-aware re-ranking
- **LLM Chatbot** — Conversational recommendations via Ollama (free, local)
- **Real-time API** — FastAPI with Prometheus metrics
- **Orchestration** — Airflow DAGs for automated pipelines
- **Monitoring** — Grafana dashboards for business & API metrics
- **Interactive UI** — Streamlit interface for exploration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI                               │
│                           (localhost:8501)                              │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                             FastAPI Server                              │
│                           (localhost:8000)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ /recommend   │  │   /chat      │  │  /feedback   │  │  /metrics   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Retrieval   │       │     Ranking     │       │   Re-ranking    │
│  (Two-Tower)  │──────▶│  (Deep Cross)   │──────▶│   (Diversity)   │
│    + FAISS    │       │                 │       │                 │
└───────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  PostgreSQL   │       │     Ollama      │       │   Prometheus    │
│  + pgvector   │       │   (LLM Chat)    │       │   + Grafana     │
└───────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for chatbot)

### 1. Clone & Setup

```bash
git clone https://github.com/rdhingana/youtube_recsys.git
cd youtube_recsys
cp .env.example .env
make install
```

### 2. Start Services

```bash
# Start PostgreSQL, monitoring, and Airflow
make start-all

# In separate terminals:
make start-api    # FastAPI  → http://localhost:8000
make start-ui     # Streamlit → http://localhost:8501
```

### 3. Run ML Pipeline

```bash
make pipeline     # load-data → generate-embeddings → build-index → train
```

### 4. (Optional) Enable Chatbot

```bash
ollama serve
ollama pull llama3.2
```

---

## 📋 Available Commands

```bash
make help         # Show all commands

# Setup
make install           # Install Python dependencies
make setup-airflow     # Initialize Airflow
make setup-monitoring  # Setup Grafana + Prometheus

# Services
make start-db          # PostgreSQL (port 5432)
make start-api         # FastAPI (port 8000)
make start-ui          # Streamlit (port 8501)
make start-airflow     # Airflow (port 8080)
make start-monitoring  # Grafana (3001) + Prometheus (9090)
make start-all         # Start all background services

make stop-all          # Stop everything
make status            # Check what's running

# ML Pipeline
make load-data              # Load videos & simulate users
make generate-embeddings    # Generate CLIP/text embeddings
make build-index            # Build FAISS index
make train                  # Train two-tower model
make pipeline               # Run full pipeline
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/recommend` | POST | Get recommendations |
| `/recommend/{user_id}` | GET | Get recommendations |
| `/chat/` | POST | Chat with AI assistant |
| `/videos/{video_id}` | GET | Video details |
| `/feedback` | POST | Submit interaction |
| `/metrics` | GET | Prometheus metrics |

**API Docs:** http://localhost:8000/docs

---

## 🎯 Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | — |
| **FastAPI** | http://localhost:8000 | — |
| **API Docs** | http://localhost:8000/docs | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **Grafana** | http://localhost:3001 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |

---

## 📁 Project Structure

```
youtube_recsys/
├── data/
│   ├── scraper/           # YouTube data collection
│   └── simulator/         # User behavior simulation
├── features/
│   ├── video_encoder.py   # CLIP + Sentence Transformers
│   └── user_encoder.py    # User embedding aggregation
├── models/
│   ├── retrieval/         # Two-tower model + FAISS
│   ├── ranking/           # Deep Cross Network
│   ├── reranking/         # Diversity optimization
│   └── pipeline.py        # End-to-end pipeline
├── serving/
│   ├── api/               # FastAPI application
│   └── chatbot/           # LLM-powered chat
├── pipelines/
│   └── dags/              # Airflow DAGs
├── monitoring/
│   ├── prometheus/        # Metrics collection
│   └── grafana/           # Dashboards
├── ui/
│   ├── app.py             # Streamlit main
│   └── pages/             # UI pages
├── scripts/               # Pipeline scripts
├── sql/                   # Database schema
├── tests/                 # Test suite
├── Makefile               # All commands
├── docker-compose.yml     # PostgreSQL
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Database** | PostgreSQL 16 + pgvector |
| **Backend** | FastAPI + Uvicorn |
| **ML Models** | PyTorch, CLIP, Sentence Transformers |
| **Vector Search** | FAISS |
| **LLM** | Ollama (Llama 3.2, Mistral) |
| **Orchestration** | Apache Airflow |
| **Monitoring** | Prometheus + Grafana |
| **UI** | Streamlit + Plotly |

---

## 📊 Airflow DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `daily_data_refresh` | 2:00 AM | Scrape new videos, simulate interactions |
| `embedding_generation` | 4:00 AM | Generate embeddings for new content |
| `model_retraining` | Sundays 6:00 AM | Retrain recommendation models |

---

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Database
POSTGRES_USER=recsys
POSTGRES_PASSWORD=recsys_password
POSTGRES_DB=youtube_recsys

# LLM (optional - Ollama auto-detected)
OPENAI_API_KEY=sk-...        # Optional
ANTHROPIC_API_KEY=sk-...     # Optional
```

---

## 🧪 Development

```bash
make test         # Run tests
make lint         # Run linter
make format       # Format code
make clean        # Clean cache files
make logs-airflow # Tail Airflow logs
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using PyTorch, FastAPI, and Streamlit
</p>