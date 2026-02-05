# YouTube RecSys - Makefile
# Production-grade video recommendation system

.PHONY: help install setup-airflow setup-monitoring \
        start-api start-ui start-airflow start-monitoring start-all \
        stop-airflow stop-monitoring stop-all status \
        build-index generate-embeddings load-data train \
        test lint clean logs

# Variables
AIRFLOW_HOME := $(PWD)/pipelines
PYTHONPATH := $(PWD):$(PYTHONPATH)

# Port Configuration
API_PORT := 8000
UI_PORT := 8501
AIRFLOW_PORT := 8080
GRAFANA_PORT := 3001
PROMETHEUS_PORT := 9090
POSTGRES_PORT := 5432

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
	@echo "  make start-db          Start PostgreSQL container"
	@echo "  make start-api         Start FastAPI server"
	@echo "  make start-ui          Start Streamlit UI"
	@echo "  make start-airflow     Start Airflow scheduler + webserver"
	@echo "  make start-monitoring  Start Grafana + Prometheus containers"
	@echo "  make start-all         Start all background services"
	@echo ""
	@echo "Stop Services:"
	@echo "  make stop-db           Stop PostgreSQL container"
	@echo "  make stop-api          Stop FastAPI server"
	@echo "  make stop-ui           Stop Streamlit UI"
	@echo "  make stop-airflow      Stop Airflow processes"
	@echo "  make stop-monitoring   Stop monitoring containers"
	@echo "  make stop-all          Stop all services"
	@echo ""
	@echo "Status:"
	@echo "  make status            Show running services and ports"
	@echo "  make ports             Show port configuration"
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
# Status & Ports
#---------------------------------------------------------------------------
ports:
	@echo ""
	@echo "Port Configuration"
	@echo "======================================"
	@echo "  PostgreSQL:   $(POSTGRES_PORT)"
	@echo "  FastAPI:      $(API_PORT)"
	@echo "  Streamlit UI: $(UI_PORT)"
	@echo "  Airflow:      $(AIRFLOW_PORT)"
	@echo "  Grafana:      $(GRAFANA_PORT)"
	@echo "  Prometheus:   $(PROMETHEUS_PORT)"
	@echo ""

status:
	@echo ""
	@echo "Service Status"
	@echo "======================================"
	@echo ""
	@echo "Docker Containers:"
	@docker ps --format "  {{.Names}}: {{.Status}}" 2>/dev/null | grep -E "(postgres|grafana|prometheus)" || echo "  No containers running"
	@echo ""
	@echo "Airflow:"
	@if pgrep -f "airflow scheduler" > /dev/null; then echo "  Scheduler: Running"; else echo "  Scheduler: Stopped"; fi
	@if pgrep -f "airflow webserver" > /dev/null; then echo "  Webserver: Running"; else echo "  Webserver: Stopped"; fi
	@echo ""
	@echo "Active Ports:"
	@echo "  PostgreSQL ($(POSTGRES_PORT)):   $$(lsof -i :$(POSTGRES_PORT) -sTCP:LISTEN -t > /dev/null 2>&1 && echo 'LISTENING' || echo 'FREE')"
	@echo "  FastAPI ($(API_PORT)):       $$(lsof -i :$(API_PORT) -sTCP:LISTEN -t > /dev/null 2>&1 && echo 'LISTENING' || echo 'FREE')"
	@echo "  Streamlit ($(UI_PORT)):     $$(lsof -i :$(UI_PORT) -sTCP:LISTEN -t > /dev/null 2>&1 && echo 'LISTENING' || echo 'FREE')"
	@echo "  Airflow ($(AIRFLOW_PORT)):      $$(lsof -i :$(AIRFLOW_PORT) -sTCP:LISTEN -t > /dev/null 2>&1 && echo 'LISTENING' || echo 'FREE')"
	@echo "  Grafana ($(GRAFANA_PORT)):      $$(lsof -i :$(GRAFANA_PORT) -sTCP:LISTEN -t > /dev/null 2>&1 && echo 'LISTENING' || echo 'FREE')"
	@echo "  Prometheus ($(PROMETHEUS_PORT)):   $$(lsof -i :$(PROMETHEUS_PORT) -sTCP:LISTEN -t > /dev/null 2>&1 && echo 'LISTENING' || echo 'FREE')"
	@echo ""
	@echo "URLs (when running):"
	@echo "  API:        http://localhost:$(API_PORT)"
	@echo "  API Docs:   http://localhost:$(API_PORT)/docs"
	@echo "  Streamlit:  http://localhost:$(UI_PORT)"
	@echo "  Airflow:    http://localhost:$(AIRFLOW_PORT)  (admin/admin)"
	@echo "  Grafana:    http://localhost:$(GRAFANA_PORT)  (admin/admin)"
	@echo "  Prometheus: http://localhost:$(PROMETHEUS_PORT)"
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
	@echo ""
	@echo "Airflow setup complete!"
	@echo "  URL: http://localhost:$(AIRFLOW_PORT)"
	@echo "  Credentials: admin/admin"

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
start-db:
	@echo "Starting PostgreSQL..."
	docker-compose up -d
	@echo ""
	@echo "PostgreSQL started!"
	@echo "  Host: localhost:$(POSTGRES_PORT)"
	@echo "  Database: youtube_recsys"
	@echo "  User: recsys"

start-api:
	@echo "Starting FastAPI server on port $(API_PORT)..."
	uvicorn serving.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

start-api-bg:
	@echo "Starting FastAPI server (background)..."
	@mkdir -p logs
	nohup uvicorn serving.api.main:app --host 0.0.0.0 --port $(API_PORT) > logs/api.log 2>&1 &
	@echo ""
	@echo "FastAPI started!"
	@echo "  URL: http://localhost:$(API_PORT)"
	@echo "  Docs: http://localhost:$(API_PORT)/docs"

start-ui:
	@echo "Starting Streamlit UI on port $(UI_PORT)..."
	streamlit run ui/app.py --server.port $(UI_PORT) --server.address 0.0.0.0

start-ui-bg:
	@echo "Starting Streamlit UI (background)..."
	@mkdir -p logs
	nohup streamlit run ui/app.py --server.port $(UI_PORT) --server.address 0.0.0.0 > logs/ui.log 2>&1 &
	@echo ""
	@echo "Streamlit started!"
	@echo "  URL: http://localhost:$(UI_PORT)"

start-airflow:
	@echo "Starting Airflow..."
	@test -f "$(AIRFLOW_HOME)/airflow.db" || (echo "Airflow not initialized. Run 'make setup-airflow' first" && exit 1)
	-pkill -f "airflow scheduler" 2>/dev/null || true
	-pkill -f "airflow webserver" 2>/dev/null || true
	@sleep 2
	AIRFLOW_HOME=$(AIRFLOW_HOME) nohup airflow scheduler > $(AIRFLOW_HOME)/logs/scheduler.log 2>&1 &
	AIRFLOW_HOME=$(AIRFLOW_HOME) nohup airflow webserver --port $(AIRFLOW_PORT) > $(AIRFLOW_HOME)/logs/webserver.log 2>&1 &
	@echo ""
	@echo "Airflow started!"
	@echo "  URL: http://localhost:$(AIRFLOW_PORT)"
	@echo "  Credentials: admin/admin"
	@echo "  Logs: make logs-airflow"

start-monitoring:
	@echo "Starting monitoring containers..."
	cd monitoring && docker-compose up -d
	@echo ""
	@echo "Monitoring started!"
	@echo "  Grafana:    http://localhost:$(GRAFANA_PORT)  (admin/admin)"
	@echo "  Prometheus: http://localhost:$(PROMETHEUS_PORT)"

start-all: start-db start-monitoring start-airflow
	@echo ""
	@echo "======================================"
	@echo "Background services started!"
	@echo "======================================"
	@echo ""
	@echo "Running services:"
	@echo "  PostgreSQL: localhost:$(POSTGRES_PORT)"
	@echo "  Grafana:    http://localhost:$(GRAFANA_PORT)  (admin/admin)"
	@echo "  Prometheus: http://localhost:$(PROMETHEUS_PORT)"
	@echo "  Airflow:    http://localhost:$(AIRFLOW_PORT)  (admin/admin)"
	@echo ""
	@echo "Start these in separate terminals:"
	@echo "  make start-api   -> http://localhost:$(API_PORT)"
	@echo "  make start-ui    -> http://localhost:$(UI_PORT)"
	@echo ""
	@echo "Check status: make status"
	@echo ""



#---------------------------------------------------------------------------
# Stop Services
#---------------------------------------------------------------------------
stop-db:
	@echo "Stopping PostgreSQL..."
	-docker-compose down 2>/dev/null || true
	@echo "PostgreSQL stopped."

stop-api:
	@echo "Stopping FastAPI..."
	-pkill -f "uvicorn.*serving.api" 2>/dev/null || true
	-fuser -k $(API_PORT)/tcp 2>/dev/null || true
	@echo "FastAPI stopped."

stop-ui:
	@echo "Stopping Streamlit..."
	-pkill -f "streamlit.*ui/app.py" 2>/dev/null || true
	-fuser -k $(UI_PORT)/tcp 2>/dev/null || true
	@echo "Streamlit stopped."

stop-airflow:
	@echo "Stopping Airflow..."
	-pkill -f "airflow scheduler" 2>/dev/null || true
	-pkill -f "airflow webserver" 2>/dev/null || true
	@echo "Airflow stopped."

stop-monitoring:
	@echo "Stopping monitoring containers..."
	-cd monitoring && docker-compose down 2>/dev/null || true
	@echo "Monitoring stopped."

stop-all: stop-api stop-ui stop-airflow stop-monitoring stop-db
	@echo ""
	@echo "======================================"
	@echo "All services stopped."
	@echo "======================================"


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