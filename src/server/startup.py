#!/usr/bin/env python3
"""
Start the HVAC Controller API server
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import config, create_default_config_file


def main():
    """Start the server with configuration"""
    parser = argparse.ArgumentParser(description="Start HVAC Controller API Server")
    
    parser.add_argument("--config-file", default="./config/hvac_config.json", 
                       help="Configuration file path (default: ./config/hvac_config.json)")
    parser.add_argument("--host", default=os.getenv("HVAC_HOST", "0.0.0.0"), help="Server host")
    parser.add_argument("--port", type=int, default=int(os.getenv("HVAC_PORT", "8000")), help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Configuration file {args.config_file} not found.")
        
        # Create config directory if needed
        config_dir = config_path.parent
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy from default config
        default_config_path = Path("hvac_mpc_controller/default_hvac_config.json")
        if default_config_path.exists():
            shutil.copy2(default_config_path, args.config_file)
            print(f"Created {args.config_file} from default template.")
        else:
            create_default_config_file(str(args.config_file))
            print(f"Created {args.config_file} with basic configuration.")
    
    # Load configuration
    try:
        full_config = config.load_config(args.config_file)
        print(f"Loaded configuration from {args.config_file}")
        print(f"Building: {len(full_config.building.walls)} walls, "
              f"{len(full_config.building.windows)} windows")
        print(f"Controller: {full_config.controller.horizon_hours}h horizon, "
              f"{full_config.controller.step_size_hours}h steps")
        
        # Use server config from file, with fallback to command line args
        server_host = args.host if args.host != os.getenv("HVAC_HOST", "0.0.0.0") else full_config.server.host
        server_port = args.port if args.port != int(os.getenv("HVAC_PORT", "8000")) else full_config.server.port
        server_reload = args.reload or full_config.server.reload
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Start server
    try:
        import uvicorn
        from server.main import app
        
        print(f"Starting server on {server_host}:{server_port}")
        print(f"API documentation available at: http://localhost:{server_port}/docs")
        
        uvicorn.run(
            app,
            host=server_host,
            port=server_port,
            reload=server_reload
        )
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install server dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error starting server: {e}")


if __name__ == "__main__":
    main() 