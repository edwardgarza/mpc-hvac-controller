#!/usr/bin/env python3
"""
Automation runner for HVAC controller
Fetches sensor data and makes predictions
"""

import json
import requests
import logging
import time
from typing import Dict, Any, Optional
from sensor_fetcher import HomeAssistantSensorFetcher

logger = logging.getLogger(__name__)

class HVACAutomationRunner:
    """Runs HVAC automation by fetching sensor data and making predictions"""
    
    def __init__(self, addon_url: str = "http://localhost:8000"):
        self.addon_url = addon_url
        self.sensor_fetcher = HomeAssistantSensorFetcher()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from add-on options"""
        try:
            # Read add-on configuration
            with open("/data/options.json", "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return defaults
            return {
                "co2_sensor": "sensor.living_room_co2",
                "temperature_sensor": "sensor.living_room_temperature", 
                "weather_entity": "weather.home",
                "horizon_hours": 24
            }
    
    def make_prediction_request(self, sensor_data) -> Dict[str, Any]:
        """Format sensor data into a prediction request"""
        return {
            "current_co2_ppm": sensor_data.co2_ppm,
            "current_temp_c": sensor_data.temperature_c,
            "current_time_hours": 0,  # Could calculate from current time
            "weather_data": sensor_data.weather_forecast,
            "horizon_hours": self.load_config().get("horizon_hours", 24)
        }
    
    def call_prediction_api(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call the prediction API endpoint"""
        try:
            url = f"{self.addon_url}/predict"
            response = requests.post(url, json=request_data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call prediction API: {e}")
            return None
    
    def run_automation(self):
        """Run the complete automation cycle"""
        logger.info("Starting HVAC automation cycle")
        
        # Load configuration
        config = self.load_config()
        logger.info(f"Loaded config: CO2 sensor={config['co2_sensor']}, "
                   f"Temp sensor={config['temperature_sensor']}, "
                   f"Weather={config['weather_entity']}")
        
        # Fetch sensor data
        sensor_data = self.sensor_fetcher.fetch_sensor_data(
            co2_sensor=config["co2_sensor"],
            temp_sensor=config["temperature_sensor"], 
            weather_entity=config["weather_entity"]
        )
        
        # Format prediction request
        request_data = self.make_prediction_request(sensor_data)
        
        # Call prediction API
        prediction = self.call_prediction_api(request_data)
        
        if prediction:
            logger.info(f"Prediction successful: CO2 trajectory length={len(prediction.get('co2_trajectory', []))}")
            logger.info(f"Next prediction: {prediction.get('next_prediction')}")
            
            # Here you could:
            # 1. Store the prediction in Home Assistant
            # 2. Trigger automations based on predictions
            # 3. Send notifications
            # 4. Update HVAC controls
            
            return prediction
        else:
            logger.error("Prediction failed")
            return None

def main():
    """Main entry point for automation"""
    logging.basicConfig(level=logging.INFO)
    
    runner = HVACAutomationRunner()
    result = runner.run_automation()
    
    if result:
        print("Automation completed successfully")
        print(f"Prediction: {result.get('next_prediction')}")
    else:
        print("Automation failed")
        exit(1)

if __name__ == "__main__":
    main() 