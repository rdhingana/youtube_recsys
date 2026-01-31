# YouTube-Style Video Recommendation System

A production-grade video recommendation system using two-tower architecture with retrieval, ranking, and re-ranking stages.

## Project Status

- [x] Phase 1: Initial project setup with PostgreSQL schema
- [ ] Phase 2: Data pipeline (scraper + simulator)
- [ ] Phase 3: Feature engineering (CLIP + transformers)
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

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/youtube-recsys.git
cd youtube-recsys
```

### 2. Setup environment

```bash
cp .env.example .env
```

### 3. Start PostgreSQL

```bash
docker-compose up -d
```

### 4. Verify database

```bash
docker exec -it youtube-recsys-db psql -U recsys -d youtube_recsys -c "\dt"
```

You should see the tables: `videos`, `users`, `user_interactions`, etc.

## Project Structure

```
youtube-recsys/
├── sql/
│   └── schema.sql          # Database schema
├── data/                   # Data files (Phase 2)
├── features/               # Feature engineering (Phase 3)
├── models/                 # ML models (Phase 4-5)
├── serving/                # FastAPI app (Phase 6)
├── ui/                     # Streamlit app (Phase 10)
├── pipelines/              # Airflow DAGs (Phase 8)
├── monitoring/             # Grafana dashboards (Phase 9)
├── tests/                  # Tests
├── notebooks/              # Jupyter notebooks
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT