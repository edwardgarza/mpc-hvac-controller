#!/bin/bash

echo "Starting HVAC Controller add-on"

# Debug: check what files are available
echo "Files in current directory:"
ls -la

if [ -f /data/options.json ]; then
  cat /data/options.json
else
  echo "(no /data/options.json present)"
fi

echo "---- bashio::jq '.' ----"
bashio::jq '.' || echo "(bashio::jq failed)"

PORT=$(bashio::config 'port')
# Start the server using the existing start_server.py
echo "Starting server on 0.0.0.0:${PORT}"
exec python3 start_server.py --host 0.0.0.0 --port 8000 --config-file /data/hvac_config.json