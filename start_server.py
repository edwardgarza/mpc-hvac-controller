#!/usr/bin/env python3
"""
Entry point for starting the HVAC Controller API server
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set up the package structure
import server.startup

if __name__ == "__main__":
    server.startup.main() 