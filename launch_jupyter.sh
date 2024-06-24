#!/bin/bash

# Check if a Jupyter Lab instance is already running and kill it
PIDS=$(ps aux | grep '[j]upyter-lab' | awk '{print $2}')
if [ ! -z "$PIDS" ]; then
  echo "Found existing Jupyter Lab process(es): $PIDS"
  echo "Killing existing Jupyter Lab process(es)..."
  kill $PIDS
fi

# Launch Jupyter Lab
echo "Launching Jupyter Lab..."
nohup jupyter lab --no-browser --port=8888 > jupyter_lab.log 2>&1 &

# Confirm that Jupyter Lab is running
sleep 2
PIDS=$(ps aux | grep '[j]upyter-lab' | awk '{print $2}')
if [ ! -z "$PIDS" ]; then
  echo "Jupyter Lab is running with PID: $PIDS"
else
  echo "Failed to start Jupyter Lab."
fi
