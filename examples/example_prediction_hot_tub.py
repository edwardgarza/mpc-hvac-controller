#!/usr/bin/env python3
"""
Example demonstrating the integrated HVAC controller with TimeSeries weather data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dateutil.parser
import numpy as np
from src.controllers.ventilation.models import RoomCO2Dynamics

from src.models.building import BuildingModel, WallModel, RoofModel, PierAndBeam, Studs
from src.models.thermal_device import HeatPumpThermalDeviceModel, ElectricResistanceThermalDeviceModel
from src.controllers.hvac_controller import HvacController
from src.models.weather import WeatherConditions, SolarIrradiation
from src.utils.orientation import Orientation
from src.utils.timeseries import TimeSeries


def create_example_room():
    
    # no co2 in a hot tub
    room_dynamics = RoomCO2Dynamics(
        volume_m3=100.0,
        controllable_ventilations=[],
        natural_ventilations=[],
    )
    
    return room_dynamics


def create_example_hot_tub_building():
    # 9x9x4 ft hot tub with 400 gallons, or about 3x3x1 meters
    wall = WallModel(Studs(0.038, 0.089, 0.406), 3, 12, Orientation())
    roof = RoofModel(3, 9, Orientation(), 0.85)  
    floor = PierAndBeam(Studs(0.038, 0.089, 0.406), 2, 9, Orientation())
    heating_model = ElectricResistanceThermalDeviceModel((0, 4000))
    building_model = BuildingModel(
        thermal_models=[wall, roof, floor],
        heating_model=heating_model,
        heat_capacity=6.3 * 10 ** 6,
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
            ground_temperature=12.0
        )
        
        weather_conditions.append(weather)
    
    # Create TimeSeries
    weather_series = TimeSeries(time_points, weather_conditions)
    return weather_series


def run_hot_tub_example():
    """Run the HVAC controller example a hot tub"""
    
    print("Setting up hot tub HVAC controller...")
    
    room_dynamics = create_example_room()
    building_model = create_example_hot_tub_building()
    energy_cost_per_kwh = 0.15
    
    # Create controller and ignore all co2
    controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=24.0,
        co2_weight=0.0,
        energy_weight=30.0,
        comfort_weight=3000,
        step_size_hours=0.5,
        optimization_method="SLSQP",
        max_iterations=500,
        use_boolean_occupant_comfort =True,
        use_soft_boundary_condition=False,
        co2_m3_per_hr_per_occupant=0,
        base_load_heat_w_per_occupant=-20000 # emulate using the tub
    )

    # assume hot tub is only occupied between 21:30 and 22
    controller.set_saved_schedule({"monday": [
        {"time": "09:00", "co2": 400, "temperature": 39, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},
        {"time": "14:00", "co2": 400, "temperature": 39, "energy_cost": energy_cost_per_kwh * 10, "occupancy_count": 0},
        {"time": "17:00", "co2": 400, "temperature": 39, "energy_cost": energy_cost_per_kwh * 2, "occupancy_count": 0},
        {"time": "21:30", "co2": 400, "temperature": 39, "energy_cost": energy_cost_per_kwh * 2, "occupancy_count": 1},
        {"time": "22:00", "co2": 400, "temperature": 39, "energy_cost": energy_cost_per_kwh * 2, "occupancy_count": 0}
        
        ]})

    current_co2_ppm = 400.0
    current_temp_c = 39
    
    weather_series = create_weather_timeseries(48)  # 48-hour forecast
    # pick random starting time
    start_time = dateutil.parser.isoparse("2024-01-15T09:30:00Z")

    ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
        current_co2_ppm, current_temp_c, 0, weather_series, start_time)
    outputs = controller.get_structured_controls_next_step()
    print("Total cost to heat $", outputs["estimated_cost"])
    controller.generate_plot()    
    return


if __name__ == "__main__":
    run_hot_tub_example() 