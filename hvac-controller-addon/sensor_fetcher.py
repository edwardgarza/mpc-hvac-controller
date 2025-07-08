#!/usr/bin/env python3
"""
Sensor data fetcher for Home Assistant integration
"""

import os
import requests
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Container for sensor data"""
    co2_ppm: float
    temperature_c: float
    weather_forecast: List[Dict[str, Any]]

class HomeAssistantSensorFetcher:
    """Fetches sensor data from Home Assistant"""
    
    def __init__(self):
        self.supervisor_token = os.getenv("SUPERVISOR_TOKEN")
        self.api_url = "http://supervisor/core/api"
        
        if not self.supervisor_token:
            logger.warning("SUPERVISOR_TOKEN not found - running outside Home Assistant")
            self.api_url = None
    
    def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of an entity"""
        if not self.api_url or not self.supervisor_token:
            logger.warning(f"Cannot fetch {entity_id} - not running in Home Assistant")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.supervisor_token}",
                "Content-Type": "application/json",
            }
            url = f"{self.api_url}/states/{entity_id}"
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {entity_id}: {e}")
            return None
    
    def get_sensor_value(self, entity_id: str, default: float = 0.0) -> float:
        """Get numeric value from a sensor entity"""
        state = self.get_entity_state(entity_id)
        if state and state.get("state"):
            try:
                return float(state["state"])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {entity_id} state '{state['state']}' to float")
        
        logger.warning(f"Using default value {default} for {entity_id}")
        return default
    
    def get_weather_forecast(self, weather_entity_id: str) -> List[Dict[str, Any]]:
        """Get weather forecast from a weather entity"""
        state = self.get_entity_state(weather_entity_id)
        if not state:
            logger.warning(f"Could not fetch weather entity {weather_entity_id}")
            return self._get_default_weather_forecast()
        
        forecast = state.get("attributes", {}).get("forecast", [])
        if not forecast:
            logger.warning(f"No forecast data in {weather_entity_id}")
            return self._get_default_weather_forecast()
        
        # Convert Home Assistant forecast format to our API format
        converted_forecast = []
        for i, item in enumerate(forecast[:24]):  # Limit to 24 hours
            converted_item = {
                "hour": i,
                "outdoor_temperature": item.get("temperature", 20.0),
                "wind_speed": item.get("wind_speed", 5.0),
                "solar_altitude_rad": 0.5,  # Default values
                "solar_azimuth_rad": 0.0,
                "solar_intensity_w": 800.0,
                "ground_temperature": 12.0
            }
            converted_forecast.append(converted_item)
        
        logger.info(f"Converted {len(converted_forecast)} weather forecast points")
        return converted_forecast
    
    def _get_default_weather_forecast(self) -> List[Dict[str, Any]]:
        """Return a default weather forecast when real data is unavailable"""
        logger.info("Using default weather forecast")
        return [
            {
                "hour": i,
                "outdoor_temperature": 20.0,
                "wind_speed": 5.0,
                "solar_altitude_rad": 0.5,
                "solar_azimuth_rad": 0.0,
                "solar_intensity_w": 800.0,
                "ground_temperature": 12.0
            }
            for i in range(24)
        ]
    
    def fetch_sensor_data(self, co2_sensor: str, temp_sensor: str, weather_entity: str) -> SensorData:
        """Fetch all sensor data needed for prediction"""
        co2_ppm = self.get_sensor_value(co2_sensor, 800.0)
        temp_c = self.get_sensor_value(temp_sensor, 22.0)
        weather_forecast = self.get_weather_forecast(weather_entity)
        
        logger.info(f"Fetched sensor data: CO2={co2_ppm}ppm, Temp={temp_c}°C, "
                   f"Weather points={len(weather_forecast)}")
        
        return SensorData(
            co2_ppm=co2_ppm,
            temperature_c=temp_c,
            weather_forecast=weather_forecast
        ) 