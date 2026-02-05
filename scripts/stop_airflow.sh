#!/bin/bash
# Stop Airflow

export AIRFLOW_HOME=$(pwd)/pipelines

echo "Stopping Airflow..."

# Kill using PIDs if available
if [ -f "$AIRFLOW_HOME/scheduler.pid" ]; then
    kill $(cat $AIRFLOW_HOME/scheduler.pid) 2>/dev/null || true
    rm $AIRFLOW_HOME/scheduler.pid
fi

if [ -f "$AIRFLOW_HOME/webserver.pid" ]; then
    kill $(cat $AIRFLOW_HOME/webserver.pid) 2>/dev/null || true
    rm $AIRFLOW_HOME/webserver.pid
fi

# Also kill any remaining processes
pkill -f "airflow scheduler" 2>/dev/null || true
pkill -f "airflow webserver" 2>/dev/null || true

echo "Airflow stopped."