# YouTube-Style Video Recommendation System

A production-grade video recommendation system using two-tower architecture with retrieval, ranking, and re-ranking stages.

## Project Status

- [x] Phase 1: Initial project setup with PostgreSQL schema
- [x] Phase 2: Data pipeline (scraper + simulator)
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
├── scripts/
│   └── load_data.py            # Load data into PostgreSQL
├── features/                   # Feature engineering (Phase 3)
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