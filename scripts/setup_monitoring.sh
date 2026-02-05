#!/bin/bash
# Monitoring Setup Script for WSL
# Grafana with PostgreSQL (business metrics) + Prometheus (API metrics)

set -e

echo "============================================"
echo "YouTube RecSys - Monitoring Setup"
echo "============================================"
echo ""
echo "This will setup:"
echo "  - Grafana (dashboards)"
echo "  - PostgreSQL connection (business metrics)"
echo "  - Prometheus (real-time API metrics)"
echo ""

# Create directories
mkdir -p monitoring/grafana/data
mkdir -p monitoring/prometheus/data

echo "Creating Docker Compose configuration..."

cat > monitoring/docker-compose.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
    depends_on:
      - prometheus
EOF

# Update prometheus config
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'recsys-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF

echo "Starting services..."
cd monitoring
docker-compose up -d
cd ..

echo ""
echo "============================================"
echo "Monitoring Setup Complete!"
echo "============================================"
echo ""
echo "Access URLs:"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo "  API Metrics: http://localhost:8000/metrics"
echo ""
echo "============================================"
echo "NEXT STEPS - Configure Grafana Datasources"
echo "============================================"
echo ""
echo "1. Open Grafana: http://localhost:3000"
echo "2. Login: admin / admin"
echo ""
echo "3. Add PostgreSQL datasource:"
echo "   - Go to: Connections > Data Sources > Add"
echo "   - Select: PostgreSQL"
echo "   - Host: host.docker.internal:5432"
echo "   - Database: youtube_recsys"
echo "   - User: recsys"
echo "   - Password: recsys_password"
echo "   - TLS/SSL Mode: disable"
echo "   - Click: Save & Test"
echo ""
echo "4. Add Prometheus datasource:"
echo "   - Go to: Connections > Data Sources > Add"
echo "   - Select: Prometheus"
echo "   - URL: http://prometheus:9090"
echo "   - Click: Save & Test"
echo ""
echo "5. Import dashboards:"
echo "   - Go to: Dashboards > Import"
echo "   - Upload: monitoring/grafana/dashboards/postgres_dashboard.json"
echo "   - Upload: monitoring/grafana/dashboards/api_metrics_dashboard.json"
echo ""
echo "Make sure your API is running:"
echo "  uvicorn serving.api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Stop monitoring:"
echo "  cd monitoring && docker-compose down"
echo ""