"""
Shared test fixtures and configuration for pytest
"""

import sys
from pathlib import Path
import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing"""
    from models.weather import WeatherConditions, SolarIrradiation
    
    solar = SolarIrradiation(
        altitude_rad=0.5,
        azimuth_rad=0.0,
        intensity_w=800.0
    )
    
    return WeatherConditions(
        irradiation=solar,
        wind_speed=5.0,
        outdoor_temperature=15.0,
        ground_temperature=12.0
    )

@pytest.fixture
def sample_building_config():
    """Sample building configuration for testing"""
    from utils.config import BuildingConfig, WallConfig, WindowConfig
    
    return BuildingConfig(
        walls=[WallConfig()],
        windows=[WindowConfig()],
        room=BuildingConfig().room
    )

@pytest.fixture
def sample_controller_config():
    """Sample controller configuration for testing"""
    from utils.config import ControllerConfig
    
    return ControllerConfig(
        horizon_hours=24.0,
        step_size_hours=0.25
    ) 