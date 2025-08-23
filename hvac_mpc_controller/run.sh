#!/bin/bash

echo "Starting HVAC Controller add-on"

# Debug: check what files are available
echo "Files in current directory:"
ls -la
PORT=$(bashio::config 'port')
# Start the server using the existing start_server.py
echo "Starting server on 0.0.0.0:${PORT}"
exec python3 start_server.py --host 0.0.0.0 --port $PORT --config-file /data/hvac_config.json