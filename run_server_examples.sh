#!/bin/bash
# Examples of different ways to configure the HVAC server port

echo "=== HVAC Controller Server Port Configuration Examples ==="
echo

echo "1. Default port (8000):"
echo "   python start_server.py"
echo

echo "2. Command line argument:"
echo "   python start_server.py --port 8080"
echo "   python start_server.py --host 127.0.0.1 --port 9000"
echo

echo "3. Environment variables:"
echo "   export HVAC_PORT=8080"
echo "   export HVAC_HOST=127.0.0.1"
echo "   python start_server.py"
echo

echo "4. Configuration file (hvac_config.json):"
echo "   Add server section to hvac_config.json:"
echo "   {"
echo "     \"server\": {"
echo "       \"host\": \"0.0.0.0\","
echo "       \"port\": 8080,"
echo "       \"reload\": true,"
echo "       \"log_level\": \"info\""
echo "     }"
echo "   }"
echo

echo "5. Priority order (highest to lowest):"
echo "   1. Command line arguments (--port, --host)"
echo "   2. Environment variables (HVAC_PORT, HVAC_HOST)"
echo "   3. Configuration file (hvac_config.json)"
echo "   4. Default values (port: 8000, host: 0.0.0.0)"
echo

echo "6. Docker with custom port:"
echo "   docker run -p 8080:8080 -e HVAC_PORT=8080 hvac-controller"
echo "   docker run -p 9000:9000 -v \$(pwd)/hvac_config.json:/app/hvac_config.json hvac-controller"
echo

echo "7. Test current configuration:"
echo "   curl http://localhost:8000/config | grep -o '\"port\":[0-9]*'"
echo 