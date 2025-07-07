#!/usr/bin/env python3
"""
Start the HVAC Controller API server
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import config, create_default_config_file


def main():
    """Start the server with configuration"""
    parser = argparse.ArgumentParser(description="Start HVAC Controller API Server")
    parser.add_argument("--config-file", default="hvac_config.json", 
                       help="Configuration file path (default: hvac_config.json)")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file and exit")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create config file if requested
    if args.create_config:
        create_default_config_file(args.config_file)
        print(f"Created default configuration file: {args.config_file}")
        print("Edit this file to customize your building and controller settings.")
        return
    
    # Check if config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Configuration file {args.config_file} not found.")
        print("Creating default configuration file...")
        create_default_config_file(args.config_file)
        print(f"Created {args.config_file}. Edit it to customize your settings.")
        print("Run again to start the server.")
        return
    
    # Load configuration
    try:
        full_config = config.load_config(args.config_file)
        print(f"Loaded configuration from {args.config_file}")
        print(f"Building: {len(full_config.building.walls)} walls, "
              f"{len(full_config.building.windows)} windows")
        print(f"Controller: {full_config.controller.horizon_hours}h horizon, "
              f"{full_config.controller.step_size_hours}h steps")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Start server
    try:
        import uvicorn
        from server.main import app
        
        print(f"Starting server on {args.host}:{args.port}")
        print("API documentation available at: http://localhost:8000/docs")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install server dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error starting server: {e}")


if __name__ == "__main__":
    main() 