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
    
    room_dynamics = RoomCO2Dynamics(
        volume_m3=100.0,
        controllable_ventilations=[],
        natural_ventilations=[],
    )
    
    return room_dynamics


def create_example_fridge():
    fridge_wall = RoofModel(2, 4, Orientation())
    heating_model = HeatPumpThermalDeviceModel(hspf=15, output_range=(-100, 0))
    building_model = BuildingModel(
        thermal_models=[fridge_wall],
        heating_model=heating_model,
        heat_capacity=10 ** 5, 
        baseload_interior_heating=10
    )

    return building_model

def create_weather_timeseries(forecast_hours=48):
    """Create a TimeSeries of weather conditions for the forecast period."""
    
    time_points = [float(x) for x in range(0, forecast_hours + 1)]
    weather_conditions = []
    
    for _ in time_points:
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
    
    return TimeSeries(time_points, weather_conditions)


def run_hpwh_example():
    """Run the HVAC controller example for a fridge"""
    
    print("Setting up fridge controller...")
    
    room_dynamics = create_example_room()
    building_model = create_example_fridge()
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
        base_load_heat_w_per_occupant=0 
    )
    temp_set_point = "0;4"

    # Note there really isn't a substantial benefit here - the thermal mass of the fridge isn't large enough where overcooling during cheap pricing
    # is that useful without risking freezing/spoiling food. The main benefit would likely be shifting the defrost and ice making to low cost times
    # 
    # Also the solver doesn't seem to converge to a very good solution.
    # the optimal solution is most likely to always be at the high temp set point except for right at the end of the cheaper electricity period - then 
    # the compressor should kick on at max
    controller.set_saved_schedule(
        {"monday": [
            {"time": "08:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 1},
            {"time": "20:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh / 3, "occupancy_count": 1}
        ],
        "tuesday": [
            {"time": "08:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh, "occupancy_count": 1},
            {"time": "20:00", "co2": 400, "temperature": temp_set_point, "energy_cost": energy_cost_per_kwh / 3, "occupancy_count": 1}
        ]
        }
)

    current_co2_ppm = 400.0
    current_temp_c = 2
    
    weather_series = create_weather_timeseries(48)  # 48-hour forecast
    # pick random starting time
    start_time = dateutil.parser.isoparse("2024-01-15T20:00:00Z")

    ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
        current_co2_ppm, current_temp_c, weather_series, start_time)
    outputs = controller.get_structured_controls_next_step()
    print("Total cost to cool $", outputs["estimated_cost"])
    controller.generate_plot()    
    return


if __name__ == "__main__":
    run_hpwh_example() 