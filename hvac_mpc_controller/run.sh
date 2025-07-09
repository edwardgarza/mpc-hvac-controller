#!/bin/bash

echo "Starting HVAC Controller add-on with cron automation"

# Debug: check what files are available
echo "Files in current directory:"
ls -la

# Generate config from add-on options
echo "Running generate_config.py..."
python3 generate_config.py

# Start cron daemon in background
echo "Starting cron daemon..."
crond

# Start the server using the existing start_server.py
echo "Starting server on 0.0.0.0:8000"
exec python3 start_server.py --host 0.0.0.0 --port 8000 