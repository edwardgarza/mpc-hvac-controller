#!/usr/bin/env python3
"""
Test script for the HVAC Controller API
"""

import requests
import json
import time
from typing import List

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_config():
    """Test configuration endpoints"""
    print("Testing configuration endpoints...")
    
    # Get current config
    response = requests.get(f"{BASE_URL}/config")
    print(f"Current config: {response.json()}")

    original_schedule = response.json()['schedules']['weekly_schedule']
    only_monday_schedule = {'monday': original_schedule['monday']}
    new_json = response.json()
    new_json['schedules']['weekly_schedule'] = only_monday_schedule
    requests.post(f"{BASE_URL}/config", data=json.dumps(new_json))
    time.sleep(10)
    updated_response = requests.get(f"{BASE_URL}/config")
    assert updated_response.json() == new_json
    response = requests.post(f"{BASE_URL}/config", data=response.text)
    print('response to setting new config', response.ok)

def create_sample_weather_forecast() -> List[dict]:
    """Create a sample weather forecast"""
    import numpy as np
    
    weather_data = []
    for hour in range(0, 25, 3):  # Every 3 hours for 24 hours
        # Simple sinusoidal temperature variation
        outdoor_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)
        
        weather_data.append({
            "time_hours": float(hour),
            "outdoor_temperature": outdoor_temp,
            "wind_speed": 5.0,
            "solar_intensity": 800 * max(0, np.sin(2 * np.pi * hour / 24)),
            "ground_temperature": outdoor_temp - 2
        })
    
    return weather_data


def test_control():
    """Test control endpoint"""
    print("Testing control endpoint...")
    
    weather_forecast = create_sample_weather_forecast()
    
    request_data = {
        "current_co2_ppm": 1200.0,
        "current_temp_c": 19.5,
        "weather_forecast": weather_forecast,
        "current_time_hours": 6.0,
    }
    
    response = requests.post(f"{BASE_URL}/control", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Control recommendations:")
        print(f"  Ventilation controls: {result['ventilation_controls']}")
        print(f"  HVAC control: {result['hvac_control']:.2f} kW")
        print(f"  Total cost: {result['total_cost']:.3f}")
        print(f"  Final CO2: {result['additional_info']['final_co2']:.1f} ppm")
        print(f"  Final temp: {result['additional_info']['final_temp']:.1f}°C")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
    
    print()


def test_models():
    """Test models endpoint"""
    print("Testing models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Models info: {response.json()}")
    print()


def main():
    """Run all tests"""
    print("HVAC Controller API Test Suite")
    print("=" * 40)
    
    try:
        test_health()
        test_config()
        test_models()
        test_control()
        
        print("All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main() 