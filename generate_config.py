#!/usr/bin/env python3
"""
Generate default configuration file
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.config import create_default_config_file

if __name__ == "__main__":
    create_default_config_file("hvac_config.json")
    print("Generated hvac_config.json with default settings")
    print("Edit this file to customize your building and controller configuration") 