#!/bin/bash

echo "Starting HVAC Controller add-on"

# Generate config from add-on options
python3 /app/generate_config.py

# Start the server using the existing start_server.py
cd /app
exec python3 start_server.py --host 0.0.0.0 --port 8000 