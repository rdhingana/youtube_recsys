# YouTube-Style Video Recommendation System

A production-grade video recommendation system using two-tower architecture with retrieval, ranking, and re-ranking stages.

## Project Status

- [x] Phase 1: Initial project setup with PostgreSQL schema
- [x] Phase 2: Data pipeline (scraper + simulator)
- [x] Phase 3: Feature engineering (CLIP + transformers)
- [x] Phase 4: Two-tower retrieval model with FAISS
- [x] Phase 5: Ranking and re-ranking models
- [x] Phase 6: FastAPI serving layer
- [x] Phase 7: Chatbot functionality
- [x] Phase 8: Airflow orchestration
- [x] Phase 9: Prometheus and Grafana monitoring
- [x] Phase 10: Streamlit UI
- [ ] Phase 11: Final polish

## Tech Stack

- **Database**: PostgreSQL 16 with pgvector
- **Backend**: FastAPI (coming soon)
- **ML**: PyTorch, CLIP, Sentence Transformers, FAISS
- **Orchestration**: Airflow
- **Monitoring**: Prometheus, Grafana
- **UI**: Streamlit

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- yt-dlp (for scraping)

### 1. Clone and setup

```bash
git clone https://github.com/yourusername/youtube-recsys.git
cd youtube-recsys
cp .env.example .env
```

### 2. Start PostgreSQL

```bash
sudo docker-compose up -d
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install yt-dlp (for scraping)

```bash
pip install yt-dlp
# or
sudo apt install yt-dlp
```

## Data Pipeline

### Scrape YouTube Videos

```bash
# Single search query
python data/scraper/youtube_scraper.py --mode search --query "python tutorial" --max-videos 10

# Scrape a channel
python data/scraper/youtube_scraper.py --mode channel --query "https://www.youtube.com/@Fireship" --max-videos 20

# Batch scrape multiple categories
python data/scraper/batch_scraper.py --mode all --videos-per-query 10
```

### Load Data into Database

```bash
# Load videos from JSON
python scripts/load_data.py --videos data/raw/videos_YYYYMMDD_HHMMSS.json

# Simulate users and interactions
python scripts/load_data.py --simulate-users 500 --simulate-days 30

# Check database stats
python scripts/load_data.py --stats
```

### Verify Data

```bash
sudo docker exec -it youtube-recsys-db psql -U recsys -d youtube_recsys -c "SELECT COUNT(*) FROM videos;"
sudo docker exec -it youtube-recsys-db psql -U recsys -d youtube_recsys -c "SELECT COUNT(*) FROM users;"
sudo docker exec -it youtube-recsys-db psql -U recsys -d youtube_recsys -c "SELECT COUNT(*) FROM user_interactions;"
```

## Feature Engineering

### Generate Video Embeddings

```bash
# Generate embeddings for all videos (with thumbnails - slower)
python scripts/generate_embeddings.py --videos

# Generate embeddings without thumbnails (faster)
python scripts/generate_embeddings.py --videos --skip-thumbnails

# Limit to first 100 videos
python scripts/generate_embeddings.py --videos --limit 100 --skip-thumbnails
```

### Generate User Embeddings

```bash
# Generate embeddings for all users
python scripts/generate_embeddings.py --users

# Generate all embeddings
python scripts/generate_embeddings.py --all --skip-thumbnails

# Check embedding stats
python scripts/generate_embeddings.py --stats
```

## Retrieval Model

### Build FAISS Index (Quick Start)

```bash
# Build index from embeddings (no training required)
python scripts/build_index.py --test

# Specify index type
python scripts/build_index.py --index-type IVF --test
```

### Train Two-Tower Model

```bash
# Train the two-tower retrieval model
python scripts/train_retrieval.py --epochs 10 --batch-size 256

# Evaluate only (if model already trained)
python scripts/train_retrieval.py --eval-only
```

### Test Retrieval

```python
from models.retrieval import SimpleRetrievalService

# Load service
service = SimpleRetrievalService()
service.load("models/retrieval/saved/simple_faiss_index")

# Get recommendations
video_ids, scores = service.retrieve(user_embedding, k=100)
```

## Ranking & Re-ranking

### Test Ranking Model

```bash
python -m models.ranking.ranking_model
```

### Test Re-ranking Model

```bash
python -m models.reranking.reranking_model
```

### Full Pipeline

```python
from models.pipeline import RecommendationPipeline

# Initialize pipeline
pipeline = RecommendationPipeline(
    retrieval_model_path="models/retrieval/saved/simple_faiss_index"
)

# Get recommendations for a user
recommendations = pipeline.recommend(user_id, k=20)
```

## API Server

### Start the API

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn

# Start server
python -m serving.api.main

# Or with uvicorn directly
uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/recommend` | POST | Get recommendations |
| `/recommend/{user_id}` | GET | Get recommendations (GET) |
| `/feedback` | POST | Submit user feedback |
| `/videos/{video_id}` | GET | Get video details |
| `/users/{user_id}/history` | GET | Get user watch history |

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Get stats
curl http://localhost:8000/stats

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "your-user-id", "num_recommendations": 10}'

# Get recommendations (GET)
curl "http://localhost:8000/recommend/your-user-id?n=10"

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-id", "video_id": "video-id", "interaction_type": "view", "watch_percentage": 0.8}'
```

### API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Chatbot

The chatbot provides a conversational interface for video recommendations.

### Setup LLM

**Option 1: Ollama (FREE - Recommended)**

Ollama runs open-source LLMs locally on your machine for free.

```bash
# 1. Install Ollama from https://ollama.ai

# 2. Start Ollama server
ollama serve

# 3. Pull a model (in another terminal)
ollama pull llama3.2      # Good balance of speed/quality
# OR
ollama pull llama3.2:1b   # Smaller, faster
# OR
ollama pull mistral       # Alternative good model
```

The chatbot will auto-detect Ollama and use it!

**Option 2: Paid APIs (Optional)**

Set API keys in `.env`:

```bash
# For OpenAI
OPENAI_API_KEY=your-openai-key

# Or for Anthropic
ANTHROPIC_API_KEY=your-anthropic-key
```

**Option 3: Mock (Default fallback)**

Without Ollama or API keys, the chatbot uses a mock LLM for basic responses.

### Chatbot API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/` | POST | Send a message |
| `/chat/history/{session_id}` | GET | Get chat history |
| `/chat/history/{session_id}` | DELETE | Clear session |

### Example Chat

```bash
# Start a conversation
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "your-user-id"}'

# Continue conversation (use session_id from previous response)
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend some videos", "session_id": "xxx", "user_id": "your-user-id"}'

# Search for videos
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Find videos about machine learning", "session_id": "xxx"}'
```

### Test Chatbot Locally

```bash
python -m serving.chatbot.chatbot_service
```

## Airflow Orchestration

Airflow automates the data pipelines for scraping, embedding generation, and model retraining.

### Setup Airflow (WSL)

```bash
# Install Airflow
pip install apache-airflow==2.8.0

# Make scripts executable
chmod +x scripts/setup_airflow.sh scripts/start_airflow.sh scripts/stop_airflow.sh

# Initialize Airflow (creates database and admin user)
./scripts/setup_airflow.sh
```

### Start Airflow

```bash
# Start Airflow (scheduler + webserver in background)
./scripts/start_airflow.sh
```

### Access Airflow UI

- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

### Available DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `daily_data_refresh` | 2 AM daily | Scrapes new videos, simulates interactions |
| `embedding_generation` | 4 AM daily | Generates embeddings for new data |
| `model_retraining` | 6 AM Sundays | Retrains the two-tower model |

### Manual DAG Trigger

```bash
export AIRFLOW_HOME=$(pwd)/pipelines

# Trigger a DAG manually
airflow dags trigger daily_data_refresh

# List DAGs
airflow dags list

# Check task status
airflow tasks list daily_data_refresh
```

### Stop Airflow

```bash
./scripts/stop_airflow.sh
```

## Monitoring (Grafana + PostgreSQL + Prometheus)

**Two data sources:**
- **PostgreSQL** → Business metrics (videos, users, interactions)
- **Prometheus** → Real-time API metrics (latency, request rates)

### Setup Monitoring

```bash
chmod +x scripts/setup_monitoring.sh
./scripts/setup_monitoring.sh
```

### Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| API Metrics | http://localhost:8000/metrics | - |

### Configure Datasources in Grafana

**1. PostgreSQL (for business metrics):**
- Go to: Connections → Data Sources → Add
- Select: PostgreSQL
- Host: `host.docker.internal:5432`
- Database: `youtube_recsys`
- User: `recsys` / Password: `recsys_password`
- TLS/SSL Mode: `disable`

**2. Prometheus (for API metrics):**
- Go to: Connections → Data Sources → Add
- Select: Prometheus
- URL: `http://prometheus:9090`

### Import Dashboards

| Dashboard | Source | Shows |
|-----------|--------|-------|
| `postgres_dashboard.json` | PostgreSQL | Videos, users, categories, top content |
| `api_metrics_dashboard.json` | Prometheus | Latency, request rates, feedback events |

### Stop Monitoring

```bash
cd monitoring && docker-compose down
```

## Streamlit UI

Interactive web interface for the recommendation system.

### Start the UI

```bash
# Install Streamlit
pip install streamlit plotly

# Make sure API is running first
uvicorn serving.api.main:app --host 0.0.0.0 --port 8000

# Start Streamlit (in another terminal)
chmod +x scripts/start_ui.sh
./scripts/start_ui.sh

# Or run directly
streamlit run ui/app.py
```

### Access the UI

Open http://localhost:8501 in your browser.

### Features

| Page | Description |
|------|-------------|
| **Home** | Overview and quick stats |
| **Recommendations** | Get personalized video recommendations |
| **Browse Videos** | Explore the video catalog with filters |
| **Chat** | Conversational interface with AI assistant |
| **Analytics** | Interactive charts and statistics |

### Screenshots

The UI includes:
- Video grid with thumbnails
- User selection for personalization
- Real-time recommendation timing metrics
- Interactive Plotly charts
- Category and interaction breakdown

## Project Structure

```
youtube-recsys/
├── sql/
│   └── schema.sql              # Database schema
├── data/
│   ├── scraper/
│   │   ├── youtube_scraper.py  # Single video/search scraper
│   │   └── batch_scraper.py    # Multi-category batch scraper
│   ├── simulator/
│   │   └── user_simulator.py   # User behavior simulator
│   └── raw/                    # Scraped data (gitignored)
├── features/
│   ├── video_encoder.py        # CLIP + Sentence Transformers
│   └── user_encoder.py         # User history aggregation
├── models/
│   ├── retrieval/
│   │   ├── two_tower.py        # Two-tower model architecture
│   │   ├── faiss_index.py      # FAISS index manager
│   │   └── retrieval_service.py # Retrieval service
│   ├── ranking/
│   │   └── ranking_model.py    # Deep Cross Network ranker
│   ├── reranking/
│   │   └── reranking_model.py  # Diversity & business rules
│   └── pipeline.py             # Full recommendation pipeline
├── serving/
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   └── schemas.py          # Pydantic models
│   └── chatbot/
│       ├── llm_client.py       # LLM provider clients
│       ├── chatbot_service.py  # Chatbot logic
│       └── routes.py           # Chat API routes
├── pipelines/
│   └── dags/
│       ├── daily_data_refresh.py    # Daily scraping DAG
│       ├── embedding_generation.py  # Embedding DAG
│       └── model_retraining.py      # Weekly training DAG
├── monitoring/
│   ├── metrics.py                   # Custom Prometheus metrics
│   ├── prometheus/
│   │   └── prometheus.yml           # Prometheus config
│   └── grafana/
│       └── dashboards/              # Grafana dashboards
├── ui/
│   ├── app.py                       # Main Streamlit app
│   └── pages/
│       ├── recommendations.py       # Recommendations page
│       ├── browse.py                # Browse videos page
│       ├── chat.py                  # Chat interface
│       └── analytics.py             # Analytics dashboard
├── scripts/
│   ├── load_data.py            # Load data into PostgreSQL
│   ├── generate_embeddings.py  # Generate and store embeddings
│   ├── build_index.py          # Build FAISS index
│   └── train_retrieval.py      # Train two-tower model
├── models/                     # ML models (Phase 4-5)
├── serving/                    # FastAPI app (Phase 6)
├── ui/                         # Streamlit app (Phase 10)
├── pipelines/                  # Airflow DAGs (Phase 8)
├── monitoring/                 # Grafana dashboards (Phase 9)
├── tests/                      # Tests
├── notebooks/                  # Jupyter notebooks
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT