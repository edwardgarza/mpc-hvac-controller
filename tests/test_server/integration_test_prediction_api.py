#!/usr/bin/env python3
"""
Test script for running the prediction API endpoints. Must spin up a local server first
"""

import datetime
import dateutil.parser
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
from typing import List

# Server URL
BASE_URL = "http://localhost:8000"

def create_sample_weather_data(start_time: datetime) -> List:
    """Create sample weather data for testing"""
    weather_data = []
    for hour in range(0, 40, 5):  # 0 to 24 hours
        weather_data.append({
            "time": datetime.datetime.isoformat(datetime.timedelta(hours=hour) + start_time),
            "outdoor_temperature": 15.0 + 5.0 * (hour / 6.0),  # Varying temperature
            "wind_speed": 5.0,
            "solar_altitude_rad": 0.5,
            "solar_azimuth_rad": 0.0,
            "solar_intensity_w": 800.0,
            "ground_temperature": 12.0
        })
    return weather_data

def test_plot_prediction_endpoint():
    """Test the /plot-prediction endpoint"""
    print("\nTesting /plot-prediction endpoint...")
    start_time_str = "2024-01-15T09:30:00Z"
    # First, we need to call the predict endpoint to set up the weather series
    weather_data = create_sample_weather_data(dateutil.parser.isoparse(start_time_str))
    
    predict_request_data = {
        "current_co2_ppm": 800.0,
        "current_temp_c": 22.0,
        "current_time_hours": 0.0,
        "weather_time_series": weather_data,
        "horizon_hours": 24.0,
        "current_time": start_time_str
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
    
    
    # Test plot endpoint
    plot_result = test_plot_prediction_endpoint()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Plot endpoint: {'✓' if plot_result else '✗'}") 