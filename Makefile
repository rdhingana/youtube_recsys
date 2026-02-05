# YouTube RecSys - Makefile
# Production-grade video recommendation system

.PHONY: help install setup-airflow setup-monitoring \
        start-api start-ui start-airflow start-monitoring start-all \
        stop-airflow stop-monitoring stop-all \
        build-index generate-embeddings load-data train \
        test lint clean logs

# Variables
AIRFLOW_HOME := $(PWD)/pipelines
PYTHONPATH := $(PWD):$(PYTHONPATH)
API_PORT := 8000
UI_PORT := 8501
AIRFLOW_PORT := 8080

#---------------------------------------------------------------------------
# Help
#---------------------------------------------------------------------------
help:
	@echo ""
	@echo "YouTube RecSys - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install           Install Python dependencies"
	@echo "  make setup-airflow     Initialize Airflow (DB + admin user)"
	@echo "  make setup-monitoring  Setup Grafana + Prometheus (Docker)"
	@echo "  make setup-all         Run all setup tasks"
	@echo ""
	@echo "Start Services:"
	@echo "  make start-api         Start FastAPI server (port $(API_PORT))"
	@echo "  make start-ui          Start Streamlit UI (port $(UI_PORT))"
	@echo "  make start-airflow     Start Airflow scheduler + webserver"
	@echo "  make start-monitoring  Start Grafana + Prometheus containers"
	@echo "  make start-all         Start all services"
	@echo ""
	@echo "Stop Services:"
	@echo "  make stop-airflow      Stop Airflow processes"
	@echo "  make stop-monitoring   Stop monitoring containers"
	@echo "  make stop-all          Stop all services"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make load-data         Load data into PostgreSQL"
	@echo "  make generate-embeddings  Generate CLIP/text embeddings"
	@echo "  make build-index       Build FAISS index"
	@echo "  make train             Train retrieval model"
	@echo "  make pipeline          Run full ML pipeline"
	@echo ""
	@echo "Development:"
	@echo "  make test              Run tests"
	@echo "  make lint              Run linter"
	@echo "  make logs-airflow      Tail Airflow logs"
	@echo "  make clean             Clean generated files"
	@echo ""

#---------------------------------------------------------------------------
# Setup
#---------------------------------------------------------------------------
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

setup-airflow:
	@echo "Setting up Airflow..."
	@mkdir -p $(AIRFLOW_HOME)/dags $(AIRFLOW_HOME)/logs $(AIRFLOW_HOME)/plugins
	AIRFLOW_HOME=$(AIRFLOW_HOME) airflow db init
	AIRFLOW_HOME=$(AIRFLOW_HOME) airflow users create \
		--username admin \
		--password admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com 2>/dev/null || echo "User already exists"
	@echo "Airflow setup complete!"
	@echo "  UI: http://localhost:$(AIRFLOW_PORT) (admin/admin)"

setup-monitoring:
	@echo "Setting up monitoring..."
	@mkdir -p monitoring/grafana/data monitoring/prometheus/data
	@echo "Monitoring directories created."
	@echo "Ensure monitoring/docker-compose.yml and monitoring/prometheus/prometheus.yml exist."
	@echo "Then run: make start-monitoring"

setup-all: install setup-airflow setup-monitoring
	@echo "All setup complete!"

#---------------------------------------------------------------------------
# Start Services
#---------------------------------------------------------------------------
start-api:
	@echo "Starting FastAPI server..."
	uvicorn serving.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

start-api-bg:
	@echo "Starting FastAPI server (background)..."
	@mkdir -p logs
	nohup uvicorn serving.api.main:app --host 0.0.0.0 --port $(API_PORT) > logs/api.log 2>&1 &
	@echo "API started on port $(API_PORT)"

start-ui:
	@echo "Starting Streamlit UI..."
	streamlit run ui/app.py --server.port $(UI_PORT) --server.address 0.0.0.0

start-airflow:
	@echo "Starting Airflow..."
	@test -f "$(AIRFLOW_HOME)/airflow.db" || (echo "Airflow not initialized. Run 'make setup-airflow' first" && exit 1)
	-pkill -f "airflow scheduler" 2>/dev/null || true
	-pkill -f "airflow webserver" 2>/dev/null || true
	@sleep 2
	AIRFLOW_HOME=$(AIRFLOW_HOME) nohup airflow scheduler > $(AIRFLOW_HOME)/logs/scheduler.log 2>&1 &
	AIRFLOW_HOME=$(AIRFLOW_HOME) nohup airflow webserver --port $(AIRFLOW_PORT) > $(AIRFLOW_HOME)/logs/webserver.log 2>&1 &
	@echo "Airflow started!"
	@echo "  UI: http://localhost:$(AIRFLOW_PORT) (admin/admin)"
	@echo "  Logs: make logs-airflow"

start-monitoring:
	@echo "Starting monitoring containers..."
	cd monitoring && docker-compose up -d
	@echo "Monitoring started!"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"

start-all: start-monitoring start-airflow
	@echo "Background services started!"
	@echo ""
	@echo "Now start API and UI in separate terminals:"
	@echo "  make start-api"
	@echo "  make start-ui"

#---------------------------------------------------------------------------
# Stop Services
#---------------------------------------------------------------------------
stop-airflow:
	@echo "Stopping Airflow..."
	-pkill -f "airflow scheduler" 2>/dev/null || true
	-pkill -f "airflow webserver" 2>/dev/null || true
	@echo "Airflow stopped."

stop-monitoring:
	@echo "Stopping monitoring containers..."
	-cd monitoring && docker-compose down 2>/dev/null || true
	@echo "Monitoring stopped."

stop-all: stop-airflow stop-monitoring
	@echo "All services stopped."

#---------------------------------------------------------------------------
# ML Pipeline
#---------------------------------------------------------------------------
load-data:
	@echo "Loading data into PostgreSQL..."
	python scripts/load_data.py

generate-embeddings:
	@echo "Generating embeddings..."
	python scripts/generate_embeddings.py

build-index:
	@echo "Building FAISS index..."
	python scripts/build_index.py

train:
	@echo "Training retrieval model..."
	python scripts/train_retrieval.py

pipeline: load-data generate-embeddings build-index train
	@echo "Full pipeline complete!"

#---------------------------------------------------------------------------
# Development
#---------------------------------------------------------------------------
test:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Running linter..."
	ruff check .

format:
	@echo "Formatting code..."
	ruff format .

logs-airflow:
	tail -f $(AIRFLOW_HOME)/logs/scheduler.log $(AIRFLOW_HOME)/logs/webserver.log

logs-api:
	tail -f logs/api.log

clean:
	@echo "Cleaning generated files..."
	-find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	-find . -type f -name "*.pyc" -delete 2>/dev/null || true
	-find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	-rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
	@echo "Clean complete."

#---------------------------------------------------------------------------
# Database
#---------------------------------------------------------------------------
db-shell:
	psql -h localhost -U recsys -d youtube_recsys

db-reset:
	@echo "Warning: This will reset the database!"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		psql -h localhost -U recsys -d youtube_recsys -f sql/schema.sql || \
		echo "Cancelled"