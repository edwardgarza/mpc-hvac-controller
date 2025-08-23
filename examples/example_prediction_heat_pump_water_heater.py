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

from src.models.building import BuildingModel, RoofModel
from src.models.thermal_device import HeatPumpThermalDeviceModel
from src.controllers.hvac_controller import HvacController
from src.models.weather import WeatherConditions, SolarIrradiation
from src.utils.orientation import Orientation
from src.utils.timeseries import TimeSeries


def create_example_room():
    
    room_dynamics = RoomCO2Dynamics(
        volume_m3=100.0,
        controllable_ventilations=[],
        natural_ventilations=[],
    )
    
    return room_dynamics


def create_example_water_tank():
    tankwall = RoofModel(4, 2, Orientation())
    # heating_model = ElectricResistanceThermalDeviceModel((0, 4000))
    heating_model = HeatPumpThermalDeviceModel(hspf=15, indoor_offset=3, output_range=(0, 200))
    building_model = BuildingModel(
        thermal_models=[tankwall],
        heating_model=heating_model,
        heat_capacity=6.3 * 10 ** 5, # about 40gal of water
        baseload_interior_heating=0
    )

    return building_model

def create_weather_timeseries(forecast_hours=48):
    """Create a TimeSeries of weather conditions for the forecast period."""
    
    time_points = [float(x) for x in range(0, forecast_hours + 1)]
    weather_conditions = []
    
    for hour in time_points:
        outdoor_temp = 20
        
        # Create weather conditions
        solar = SolarIrradiation(
            altitude_rad=0.0,  
            azimuth_rad=0.0,
            intensity_w=0
        )
        
        weather = WeatherConditions(
            irradiation=solar,
            wind_speed=0.0,
            outdoor_temperature=outdoor_temp,
            ground_temperature=20.0
        )
        
        weather_conditions.append(weather)
    
    # Create TimeSeries
    weather_series = TimeSeries(time_points, weather_conditions)
    return weather_series


def run_hpwh_example():
    """Run the HVAC controller example for a heat pump water heater"""
    
    print("Setting up heat pump water heater controller...")
    
    room_dynamics = create_example_room()
    building_model = create_example_water_tank()
    energy_cost_per_kwh = 0.15
    
    # Create controller and ignore all co2
    controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=24.0,
        co2_weight=0.0,
        energy_weight=300.0,
        comfort_weight=30,
        step_size_hours=0.5,
        optimization_method="SLSQP",
        max_iterations=500,
        co2_m3_per_hr_per_occupant=0,
        base_load_heat_w_per_occupant=-5000 # emulate taking a shower
    )

    # make the set point deadband pretty wide - less than 40 indicates that there may not be enough hot water 
    # and 65 is probably near the max temp a heat pump can do and can be scalding
    temp_set_point = "40;65"

    # showers taken at 8 and 18 for 30 min, but at 18 there are 2 people showering
    controller.set_saved_schedule(
        {"monday": [
            {"time": "08:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 1},
            {"time": "08:30", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},
            {"time": "18:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 2},
            {"time": "18:30", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},
            {"time": "20:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh / 3, "occupancy_count": 0}
        ],
        "tuesday": [
            {"time": "08:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 1},
            {"time": "08:30", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},
            {"time": "18:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 2},
            {"time": "18:30", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 0},
            {"time": "20:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh / 3, "occupancy_count": 0}
        ]
        }
)

    current_co2_ppm = 400.0
    current_temp_c = 39
    
    weather_series = create_weather_timeseries(48)  # 48-hour forecast
    # pick random starting time
    start_time = dateutil.parser.isoparse("2024-01-15T20:00:00Z")

    ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
        current_co2_ppm, current_temp_c, 0, weather_series, start_time)
    outputs = controller.get_structured_controls_next_step()
    print("Total cost to heat $", outputs["estimated_cost"])
    controller.generate_plot()    
    return


if __name__ == "__main__":
    run_hpwh_example() 