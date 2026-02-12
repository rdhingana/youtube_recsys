# 🎬 YouTube Video Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/PostgreSQL-16-blue.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Whisper-Speech--to--Text-orange.svg" alt="Whisper">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  A production-grade video recommendation system featuring two-tower retrieval, multi-stage ranking, LLM-powered voice chatbot, real-time feedback loop, and comprehensive monitoring.
</p>

---

## ✨ Features

### 🎯 Core ML Pipeline
- **Two-Tower Retrieval** — CLIP & Sentence Transformer embeddings with FAISS indexing
- **Multi-Stage Ranking** — Deep Cross Network (DCN) for precise scoring
- **Diversity Re-ranking** — Ensures varied, engaging recommendations

### 🖥️ Interactive UI
- **Netflix-Style Onboarding** — Pick categories, get instant recommendations
- **Real-time Feedback Loop** — 👍/👎 buttons to refine suggestions
- **Voice Chat** — Speech-to-text (Whisper) + text-to-speech
- **User Journey Analytics** — Sankey diagrams & co-watching patterns

### 🤖 AI Chatbot
- **Conversational Recommendations** — Natural language video search
- **Local LLM** — Powered by Ollama (Llama 3.2, Mistral) — free & private
- **Voice Input/Output** — Speak questions, hear responses

### 📊 Production Features
- **Real-time API** — FastAPI with sub-100ms latency
- **Prometheus Metrics** — Request tracking, latency percentiles
- **Grafana Dashboards** — Business & API monitoring
- **Airflow Orchestration** — Automated daily pipelines

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI                               │
│                           (localhost:8501)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │    Home      │  │   Recommend  │  │    Chat      │  │  Analytics  │  │
│  │  Dashboard   │  │  + Feedback  │  │  + Voice     │  │  + Sankey   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│                             FastAPI Server                              │
│                           (localhost:8000)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ /recommend   │  │   /chat      │  │  /feedback   │  │  /metrics   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐       ┌─────────────────────┐       ┌─────────────────┐
│   Retrieval   │       │      Ranking        │       │   Re-ranking    │
│  (Two-Tower)  │──────▶│   (Deep Cross)      │──────▶│   (Diversity)   │
│    + FAISS    │       │                     │       │                 │
└───────────────┘       └─────────────────────┘       └─────────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐       ┌─────────────────────┐       ┌─────────────────┐
│  PostgreSQL   │       │      Ollama         │       │   Prometheus    │
│  + pgvector   │       │   (LLM + Whisper)   │       │   + Grafana     │
└───────────────┘       └─────────────────────┘       └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- [Ollama](https://ollama.ai) (for chatbot)

### 1. Clone & Setup

```bash
git clone https://github.com/rdhingana/youtube_recsys.git
cd youtube_recsys
cp .env.example .env
make install
```

### 2. Start Services

```bash
# Start all background services (DB, monitoring, Airflow)
make start-all

# In separate terminals:
make start-api    # FastAPI  → http://localhost:8000
make start-ui     # Streamlit → http://localhost:8501
```

### 3. Run ML Pipeline

```bash
make pipeline     # load-data → generate-embeddings → build-index → train
```

### 4. Enable Voice & Chatbot

```bash
# Install voice support
pip install openai-whisper audio-recorder-streamlit

# Start Ollama for chatbot
ollama serve
ollama pull llama3.2
```

---

## 📋 Available Commands

```bash
make help              # Show all commands

# Services
make start-all         # Start PostgreSQL, Airflow, Prometheus, Grafana
make start-api         # FastAPI (port 8000)
make start-ui          # Streamlit (port 8501)
make stop-all          # Stop everything
make status            # Check what's running

# ML Pipeline
make pipeline          # Run full pipeline
make load-data         # Load videos & simulate users
make generate-embeddings
make build-index       # Build FAISS index
make train             # Train two-tower model

# Development
make test              # Run tests
make lint              # Run linter
make clean             # Clean cache files
```

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

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/recommend` | POST | Get personalized recommendations |
| `/recommend/{user_id}` | GET | Get recommendations for user |
| `/chat/` | POST | Chat with AI assistant |
| `/videos/{video_id}` | GET | Video details |
| `/feedback` | POST | Submit user feedback |
| `/metrics` | GET | Prometheus metrics |

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
│   ├── Home.py            # Streamlit main
│   └── pages/
│       ├── 1_🎯_Recommendations.py  
│       ├── 2_🔍_Browse.py
│       ├── 3_💬_Chat.py            
│       └── 4_📊_Analytics.py        
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
| **Speech-to-Text** | OpenAI Whisper (local) |
| **Text-to-Speech** | Web Speech API |
| **Orchestration** | Apache Airflow |
| **Monitoring** | Prometheus + Grafana |
| **UI** | Streamlit + Plotly |

---

## 📊 UI Features

### 🎯 Recommendations Page
- **Existing User Mode** — Select user profile, get personalized recommendations
- **Quick Start (Guest)** — Netflix-style category picker for new users
- **Feedback Loop** — 👍/👎 buttons on every video
- **Performance Metrics** — Retrieval, ranking, re-ranking latency

### 💬 Chat Page
- **Voice Input** — Click 🎤 to speak (Whisper transcription)
- **Voice Output** — Toggle "Read responses aloud" for TTS
- **Quick Suggestions** — Pre-built prompts for common queries
- **Context-Aware** — Maintains conversation history

### 📊 Analytics Page
- **Sankey Diagram** — User journey: Persona → Category → Interaction
- **Co-Watching Patterns** — "Users who watch X also watch Y"
- **Category Distribution** — Pie chart of content
- **Engagement Metrics** — Watch completion rates

---

## 📅 Airflow DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `daily_data_refresh` | 2:00 AM | Scrape new videos, simulate interactions |
| `embedding_generation` | 4:00 AM | Generate embeddings for new content |
| `model_retraining` | Sundays 6:00 AM | Retrain recommendation models |

---

# API
API_URL=http://localhost:8000

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using PyTorch, FastAPI, Streamlit, and Whisper
</p>
