#!/usr/bin/env python3
"""
Test script for the new prediction API endpoints
"""

import requests
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io

# Server URL
BASE_URL = "http://localhost:8000"

def create_sample_weather_data():
    """Create sample weather data for testing"""
    weather_data = []
    for hour in range(0, 25, 1):  # 0 to 24 hours, every 1 hour
        weather_data.append({
            "hour": float(hour),
            "outdoor_temperature": 15.0 + 5.0 * (hour / 6.0),  # Varying temperature
            "wind_speed": 5.0,
            "solar_altitude_rad": 0.5,
            "solar_azimuth_rad": 0.0,
            "solar_intensity_w": 800.0,
            "ground_temperature": 12.0
        })
    return weather_data

def test_prediction_endpoint():
    """Test the /predict endpoint"""
    print("Testing /predict endpoint...")
    
    weather_data = create_sample_weather_data()
    
    request_data = {
        "current_co2_ppm": 800.0,
        "current_temp_c": 22.0,
        "current_time_hours": 0.0,
        "weather_data": weather_data,
        "horizon_hours": 6.0  # Test with 6-hour horizon
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=request_data)
        response.raise_for_status()
        
        result = response.json()
        print("✓ Prediction endpoint successful")
        print(f"  Has prediction: {result['has_prediction']}")
        print(f"  Time horizon: {len(result['time_horizon_hours'])} steps")
        print(f"  Weather forecast: {len(result['weather_forecast'])} points")
        
        if result['next_prediction']:
            print(f"  Next prediction length: {len(result['next_prediction'])}")
        
        # Check for new trajectory data
        if result.get('co2_trajectory'):
            print(f"  CO2 trajectory: {len(result['co2_trajectory'])} points")
            print(f"    Initial CO2: {result['co2_trajectory'][0]:.1f} ppm")
            print(f"    Final CO2: {result['co2_trajectory'][-1]:.1f} ppm")
        
        if result.get('temperature_trajectory'):
            print(f"  Temperature trajectory: {len(result['temperature_trajectory'])} points")
            print(f"    Initial temp: {result['temperature_trajectory'][0]:.1f} °C")
            print(f"    Final temp: {result['temperature_trajectory'][-1]:.1f} °C")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Prediction endpoint failed: {e}")
        return None

def test_plot_prediction_endpoint():
    """Test the /plot-prediction endpoint"""
    print("\nTesting /plot-prediction endpoint...")
    
    # First, we need to call the predict endpoint to set up the weather series
    weather_data = create_sample_weather_data()
    
    predict_request_data = {
        "current_co2_ppm": 800.0,
        "current_temp_c": 22.0,
        "current_time_hours": 0.0,
        "weather_data": weather_data,
        "horizon_hours": 6.0
    }
    
    try:
        # First call predict to set up the weather series
        predict_response = requests.post(f"{BASE_URL}/predict", json=predict_request_data)
        predict_response.raise_for_status()
        print("  ✓ Called predict endpoint to set up weather series")
        
        # Now call the plot endpoint
        response = requests.get(f"{BASE_URL}/plot-prediction")
        response.raise_for_status()
        
        result = response.json()
        print("✓ Plot prediction endpoint successful")
        
        # Save the plot
        if 'plot_data' in result:
            plot_data = base64.b64decode(result['plot_data'])
            with open('prediction_plot.png', 'wb') as f:
                f.write(plot_data)
            print("  Plot saved as 'prediction_plot.png'")
            
            # Display the plot
            img = mpimg.imread('prediction_plot.png')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Prediction Plot')
            plt.tight_layout()
            plt.show()
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Plot prediction endpoint failed: {e}")
        return None

if __name__ == "__main__":
    print("Testing HVAC Controller Prediction API")
    print("=" * 50)
    
    # Test prediction endpoint
    prediction_result = test_prediction_endpoint()
    
    # Test plot endpoint
    plot_result = test_plot_prediction_endpoint()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Prediction endpoint: {'✓' if prediction_result else '✗'}")
    print(f"  Plot endpoint: {'✓' if plot_result else '✗'}") 