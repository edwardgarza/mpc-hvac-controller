#!/usr/bin/env python3
"""
Example showing how configuration is loaded in the HVAC Controller API - Config Files Only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from src.utils.config import ControllerConfig, create_default_config_file

def demonstrate_config_loading():
    """Demonstrate config file loading"""
    
    print("HVAC Controller Configuration - Config Files Only")
    print("=" * 50)
    
    # 1. Create a default config file
    print("\n1. Creating default configuration file...")
    create_default_config_file("example_config.json")
    
    # 2. Load from file
    print("\n2. Loading configuration from file...")
    try:
        from src.utils.config import get_controller_config
        config = get_controller_config()
        print(f"   Loaded config: {config.model_dump()}")
    except Exception as e:
        print(f"   Error loading config: {e}")
    
    # 3. Show configuration file example
    print("\n3. Configuration file example (hvac_config.json):")
    example_config = {
        "horizon_hours": 24.0,
        "co2_weight": 1.0,
        "energy_weight": 2.0,
        "comfort_weight": 1.5,
        "step_size_hours": 0.25,
        "optimization_method": "SLSQP",
        "max_iterations": 500
    }
    print(json.dumps(example_config, indent=2))
    
    # 4. Show how to use different configs
    print("\n4. Using different configuration files:")
    print("   # Create a development config")
    print("   python config.py")
    print("   # Edit hvac_config.json for development settings")
    print("   # Start server: python start_server.py --config-file hvac_config.json")
    print()
    print("   # Create a production config")
    print("   cp hvac_config.json production_config.json")
    print("   # Edit production_config.json for production settings")
    print("   # Start server: python start_server.py --config-file production_config.json")


def show_server_startup():
    """Show how the server loads configuration on startup"""
    
    print("\n" + "=" * 50)
    print("Server Startup Configuration Flow")
    print("=" * 50)
    
    print("\nWhen the server starts, it follows this sequence:")
    print("1. FastAPI app starts")
    print("2. @app.on_event('startup') is called")
    print("3. create_default_models() creates room and building models")
    print("4. get_controller_config() loads configuration:")
    print("   - Tries to load from config file (if specified)")
    print("   - Falls back to default values")
    print("5. create_controller(config) creates the HVAC controller")
    print("6. Server is ready to accept requests")
    
    print("\nExample server startup output:")
    print("INFO:     Uvicorn running on http://0.0.0.0:8000")
    print("INFO:     Started server process")
    print("INFO:     Waiting for application startup.")
    print("INFO:     Application startup complete.")
    print("Loaded configuration from file: hvac_config.json")
    print("Server started with configuration: {'horizon_hours': 24.0, ...}")


def show_config_file_management():
    """Show how to manage different config files"""
    
    print("\n" + "=" * 50)
    print("Configuration File Management")
    print("=" * 50)
    
    print("\nDifferent configuration files for different purposes:")
    print()
    print("hvac_config.json (default)")
    print("- General purpose configuration")
    print("- Good starting point")
    print()
    print("development_config.json")
    print("- Shorter horizon for faster testing")
    print("- Higher CO2 tolerance")
    print("- Debug logging enabled")
    print()
    print("production_config.json")
    print("- Longer horizon for better optimization")
    print("- Strict CO2 targets")
    print("- Conservative energy settings")
    print()
    print("testing_config.json")
    print("- Very short horizon")
    print("- Relaxed constraints")
    print("- Fast optimization")


if __name__ == "__main__":
    demonstrate_config_loading()
    show_server_startup()
    show_config_file_management() 