#!/bin/bash
# Airflow Setup Script for WSL (No Docker)
# Run this to initialize Airflow locally

set -e

echo "============================================"
echo "YouTube RecSys - Airflow Setup (WSL)"
echo "============================================"

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/pipelines

echo "AIRFLOW_HOME: $AIRFLOW_HOME"

# Create necessary directories
echo "Creating directories..."
mkdir -p $AIRFLOW_HOME/dags $AIRFLOW_HOME/logs $AIRFLOW_HOME/plugins

# Install Airflow if not installed
if ! command -v airflow &> /dev/null; then
    echo "Installing Apache Airflow..."
    pip install "apache-airflow==2.8.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.0/constraints-3.11.txt"
fi

# Initialize Airflow database (SQLite by default)
echo "Initializing Airflow database..."
airflow db init

# Create admin user
echo "Creating admin user..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com 2>/dev/null || echo "User already exists"

echo ""
echo "============================================"
echo "Airflow Setup Complete!"
echo "============================================"
echo ""
echo "To start Airflow, run these in separate terminals:"
echo ""
echo "Terminal 1 (Scheduler):"
echo "  export AIRFLOW_HOME=$(pwd)/pipelines"
echo "  airflow scheduler"
echo ""
echo "Terminal 2 (Webserver):"
echo "  export AIRFLOW_HOME=$(pwd)/pipelines"
echo "  airflow webserver --port 8080"
echo ""
echo "Access Airflow UI at: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "Or use the helper script:"
echo "  ./scripts/start_airflow.sh"
echo ""