#!/bin/bash
# Start Airflow (Scheduler + Webserver)
# This runs both in background

set -e

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/pipelines
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Starting Airflow..."
echo "AIRFLOW_HOME: $AIRFLOW_HOME"

# Check if Airflow is initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Airflow not initialized. Run ./scripts/setup_airflow.sh first"
    exit 1
fi

# Kill any existing Airflow processes
pkill -f "airflow scheduler" 2>/dev/null || true
pkill -f "airflow webserver" 2>/dev/null || true

sleep 2

# Start scheduler in background
echo "Starting scheduler..."
airflow scheduler > $AIRFLOW_HOME/logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler PID: $SCHEDULER_PID"

# Start webserver in background
echo "Starting webserver..."
airflow webserver --port 8080 > $AIRFLOW_HOME/logs/webserver.log 2>&1 &
WEBSERVER_PID=$!
echo "Webserver PID: $WEBSERVER_PID"

# Save PIDs for stop script
echo $SCHEDULER_PID > $AIRFLOW_HOME/scheduler.pid
echo $WEBSERVER_PID > $AIRFLOW_HOME/webserver.pid

echo ""
echo "============================================"
echo "Airflow Started!"
echo "============================================"
echo ""
echo "Airflow UI: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "View logs:"
echo "  tail -f $AIRFLOW_HOME/logs/scheduler.log"
echo "  tail -f $AIRFLOW_HOME/logs/webserver.log"
echo ""
echo "Stop Airflow:"
echo "  ./scripts/stop_airflow.sh"
echo ""