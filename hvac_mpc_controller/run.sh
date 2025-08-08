#!/bin/bash

echo "Starting HVAC Controller add-on"

# Debug: check what files are available
echo "Files in current directory:"
ls -la

# Start the server using the existing start_server.py
echo "Starting server on 0.0.0.0:8000"
exec python3 start_server.py --host 0.0.0.0 --port 8000 