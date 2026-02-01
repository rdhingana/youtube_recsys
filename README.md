# YouTube-Style Video Recommendation System

A production-grade video recommendation system using two-tower architecture with retrieval, ranking, and re-ranking stages.

## Project Status

- [x] Phase 1: Initial project setup with PostgreSQL schema
- [x] Phase 2: Data pipeline (scraper + simulator)
- [x] Phase 3: Feature engineering (CLIP + transformers)
- [x] Phase 4: Two-tower retrieval model with FAISS
- [x] Phase 5: Ranking and re-ranking models
- [x] Phase 6: FastAPI serving layer
- [ ] Phase 4: Two-tower retrieval model with FAISS
- [ ] Phase 5: Ranking and re-ranking models
- [ ] Phase 6: FastAPI serving layer
- [ ] Phase 7: Chatbot functionality
- [ ] Phase 8: Airflow orchestration
- [ ] Phase 9: Prometheus and Grafana monitoring
- [ ] Phase 10: Streamlit UI
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
│   └── api/
│       ├── main.py             # FastAPI application
│       └── schemas.py          # Pydantic models
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