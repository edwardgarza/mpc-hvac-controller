#!/usr/bin/env python3
"""
Example demonstrating the integrated HVAC controller with TimeSeries weather data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dateutil.parser
import numpy as np
from src.controllers.ventilation.models import (
    RoomCO2Dynamics, WindowVentilationModel, HRVVentilationModel, 
    ERVVentilationModel, NaturalVentilationModel
)
from src.models.building import BuildingModel, SlabModel, WallModel, WindowModel, RoofModel, PierAndBeam, Studs
from src.models.thermal_device import HeatPumpThermalDeviceModel, ElectricResistanceThermalDeviceModel
from src.controllers.hvac_controller import HvacController
from src.models.weather import WeatherConditions, SolarIrradiation
from src.utils.orientation import Orientation
from src.utils.timeseries import TimeSeries


def create_example_room():
    """Create a simple room setup for testing"""
    
    # windows would only be used for free heating/cooling
    window_vent = WindowVentilationModel(200)
    
    # Create room dynamics
    room_dynamics = RoomCO2Dynamics(
        volume_m3=100.0,
        controllable_ventilations=[window_vent],
        natural_ventilations=[],
        outdoor_co2_ppm=400
    )
    
    return room_dynamics


def create_example_building():
    wall = WallModel(Studs(0.038, 0.089, 0.406), 3, 150, Orientation())
    roof = RoofModel(3, 75, Orientation(), 0.85) 
    
    floor = SlabModel(1, 70)  
    heating_model = HeatPumpThermalDeviceModel(output_range=(-4000, 4000))
    building_model = BuildingModel(
        thermal_models=[wall, roof, floor],
        heating_model=heating_model,
        heat_capacity=10 ** 6,
        baseload_interior_heating=0
    )

    return building_model

def create_weather_timeseries(forecast_hours=48):
    """Create a TimeSeries of weather conditions for the forecast period."""
    
    # Create time points (every 3 hours for forecast data)
    time_points = [float(x) for x in range(0, forecast_hours + 1)]
    weather_conditions = []
    
    for hour in time_points:
        # Simple sinusoidal temperature variation
        outdoor_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)
        
        # Create weather conditions
        solar = SolarIrradiation(
            altitude_rad=0.5,  # Simple fixed values
            azimuth_rad=0.0,
            intensity_w=800 * max(0, np.sin(2 * np.pi * hour / 24))
        )
        
        weather = WeatherConditions(
            irradiation=solar,
            wind_speed=5.0,
            outdoor_temperature=outdoor_temp,
            ground_temperature=17.0
        )
        
        weather_conditions.append(weather)
    
    # Create TimeSeries
    weather_series = TimeSeries(time_points, weather_conditions)
    return weather_series


def run_periodically_occupied_example():
    """Run the integrated HVAC controller example with TimeSeries weather data"""
    
    print("Setting up HVAC controller example with for a periodically occupied building...")
    
    room_dynamics = create_example_room()
    building_model = create_example_building()
    energy_cost_per_kwh = 0.15
    
    # Create controller and ignore all co2
    controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=24.0,
        co2_weight=0.0,
        energy_weight=300000.0,
        comfort_weight=1,
        step_size_hours=1.0,
        optimization_method="SLSQP",
        max_iterations=500,
    )

    # assume building is only occupied between 00 and 6
    controller.set_saved_schedule({"monday": [
        {"time": "00:00", "co2": 400, "temperature": 18, "energy_cost": energy_cost_per_kwh, "occupancy_count": 1},
        {"time": "06:00", "co2": 400, "temperature": 18, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},        
        ], 
        "tuesday": [
        {"time": "00:00", "co2": 400, "temperature": 18, "energy_cost": energy_cost_per_kwh, "occupancy_count": 1},
        {"time": "06:00", "co2": 400, "temperature": 18, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},
        
        ]})

    # Initial conditions
    current_co2_ppm = 400.0
    current_temp_c = 15
    
    # Create weather TimeSeries
    weather_series = create_weather_timeseries(48)  # 48-hour forecast
    # pick random starting time
    start_time = dateutil.parser.isoparse("2024-01-15T09:30:00Z")

    ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
        current_co2_ppm, current_temp_c, weather_series, start_time)
    outputs = controller.get_structured_controls_next_step()
    print("Total cost to heat $", outputs["estimated_cost"])
    controller.generate_plot()    
    return


if __name__ == "__main__":
    run_periodically_occupied_example() 