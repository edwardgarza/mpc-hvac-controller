#!/usr/bin/env python3
"""
Test script for the HVAC Controller add-on
"""

import os
import sys
import tempfile
from pathlib import Path
import json

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_generation():
    """Test that config generation works"""
    print("Testing config generation...")
    
    # Set some test environment variables
    os.environ["horizon_hours"] = "12"
    os.environ["co2_target_ppm"] = "600"
    os.environ["temp_target_c"] = "20.0"
    os.environ["port"] = "9000"
    
    # Import and run the config generator
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_config import generate_config_from_options
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Run the config generator
            generate_config_from_options()
            
            # Check if config file was created
            config_file = Path("hvac_config.json")
            if config_file.exists():
                print("✅ Config file generated successfully")
                
                # Read and verify the config
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                # Check that our environment variables were used
                assert config["controller"]["horizon_hours"] == 12
                assert config["controller"]["co2_target_ppm"] == 600
                assert config["controller"]["temp_target_c"] == 20.0
                assert config["server"]["port"] == 9000
                
                print("✅ Config values match environment variables")
                return True
            else:
                print("❌ Config file was not created")
                return False
                
        finally:
            os.chdir(original_cwd)

def test_dockerfile_syntax():
    """Test that Dockerfile has correct syntax"""
    print("\nTesting Dockerfile syntax...")
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    with open(dockerfile_path, "r") as f:
        content = f.read()
    
    # Check for required elements
    required = [
        "FROM",
        "WORKDIR",
        "COPY",
        "EXPOSE",
        "CMD"
    ]
    
    for req in required:
        if req in content:
            print(f"  ✅ {req} found")
        else:
            print(f"  ❌ {req} missing")
            return False
    
    print("✅ Dockerfile syntax looks good")
    return True

def main():
    """Run all tests"""
    print("🧪 Testing HVAC Controller Add-on")
    print("=" * 40)
    
    tests = [
        test_config_generation,
        test_dockerfile_syntax,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Add-on is ready.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")

if __name__ == "__main__":
    main() 