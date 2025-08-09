import dateutil.parser
import numpy as np
import unittest
from src.controllers.hvac import HvacController
from src.controllers.ventilation.models import (
    RoomCO2Dynamics, WindowVentilationModel, HRVVentilationModel, 
    ERVVentilationModel, NaturalVentilationModel, CO2Source
)
from src.models.building import BuildingModel, WallModel, WindowModel, RoofModel, PierAndBeam, Studs
from src.models.thermal_device import HeatPumpThermalDeviceModel, ElectricResistanceThermalDeviceModel
from src.controllers.hvac import HvacController
from src.models.weather import WeatherConditions, SolarIrradiation
from src.utils.orientation import Orientation
from src.utils.timeseries import TimeSeries



class TestHVACController(unittest.TestCase):

    def create_example_room():
        """Create a simple room setup for testing"""
        
        # Create ventilation models
        window_vent = WindowVentilationModel()
        erv_vent = ERVVentilationModel(heat_recovery_efficiency=0.9, fan_power_w_m3_per_hour=0.3)
        natural_vent = NaturalVentilationModel(indoor_volume_m3=100.0, infiltration_rate_ach=0.2)
        
        # Create CO2 sources (occupants)
        occupant_source = CO2Source(co2_production_rate_m3_per_hour=0.02)  # 2 people
        
        # Create room dynamics
        room_dynamics = RoomCO2Dynamics(
            volume_m3=100.0,
            sources=[occupant_source],
            controllable_ventilations=[window_vent, erv_vent],
            natural_ventilations=[natural_vent],
            outdoor_co2_ppm=400
        )
        
        return room_dynamics


    def create_example_building():
        """Create a simple building model"""
        
        # Create building components
        studs = Studs(0.038, 0.089, 0.406)
        wall = WallModel(studs, 0, 1, Orientation())
        window = WindowModel(0.7, 1, 0.7)  

        roof = RoofModel(10, 50, Orientation(), 0.85)  # 50 m² roof
        floor = PierAndBeam(studs, 5, 50, Orientation())  # 50 m² floor
        
        # Create heating model
        heating_model = HeatPumpThermalDeviceModel(hspf=9.0, output_range=(-10000, 10000))
        # heating_model = ElectricResistanceThermalDeviceModel()
        # Create building model
        building_model = BuildingModel(
            thermal_models=[wall, window, roof, floor],
            heating_model=heating_model,
            heat_capacity=10 ** 6  # J/K
        )
        
        return building_model


    def create_weather_timeseries(self, forecast_hours=48):
        """Create a TimeSeries of weather conditions for the forecast period."""
        
        # Create time points (every 3 hours for forecast data)
        time_points = [float(x) for x in range(0, forecast_hours + 1)]
        weather_conditions = []
        
        for hour in time_points:
            outdoor_temp = 15 
            
            # Create weather conditions
            solar = SolarIrradiation(
                altitude_rad=0,  # Simple fixed values
                azimuth_rad=0.0,
                intensity_w=0
            )
            
            weather = WeatherConditions(
                irradiation=solar,
                wind_speed=0.0,
                outdoor_temperature=outdoor_temp,
                ground_temperature=12.0
            )
            
            weather_conditions.append(weather)
        
        # Create TimeSeries
        weather_series = TimeSeries(time_points, weather_conditions)
        return weather_series
    room_dynamics = create_example_room()
    building_model = create_example_building()

    default_controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=12.0,
        co2_weight=0.25,
        energy_weight=100.0,
        comfort_weight=0.1,
        step_size_hours=0.5,
        optimization_method="SLSQP",
        max_iterations=500,
    )
    default_controller.set_saved_schedule({"monday": [
        {"time": "09:00", "co2": 800, "temperature": 21, "energy_cost": 0.15, "occupancy_count": 1}]})


    def test_step_sizes_static(self):
        steps, cumulative_time = self.default_controller.generate_time_steps(0.25, 12, False)
        self.assertTrue(all([x == 0.25 for x in steps]))
        self.assertEqual(len(steps), len(cumulative_time))
        self.assertLessEqual(cumulative_time[-1], 12)
        self.assertSequenceEqual([int(x / 0.25) for x in cumulative_time], list(range(1, len(cumulative_time) + 1)))

    def test_step_sizes_dynamic(self):
        steps, cumulative_time = self.default_controller.generate_time_steps(0.25, 12, True)
        self.assertTrue(all([x / 0.25 == int(x / 0.25) for x in steps]))
        self.assertEqual(len(steps), len(cumulative_time))
        self.assertLessEqual(cumulative_time[-1], 12)

    def test_get_control_info_no_exception(self):
        start_time = dateutil.parser.isoparse("2024-01-15T09:30:00Z")
        control_info = self.default_controller.get_control_info(1200, 20, self.create_weather_timeseries(), start_time)
        print(control_info)
        self.assertGreater(control_info['total_energy_cost_dollars'], 0)

        # the total energy cost should be slightly higher than the hvac usage, which is $0.15/kwh (and the step size is 0.5 hours)
        hvac_energy_used = 0
        for hvac_input in control_info['hvac_controls']:
            hvac_energy_used += sum([abs(x) for x in hvac_input])
        self.assertGreater(control_info['total_energy_cost_dollars'], hvac_energy_used * 0.15 / 1000 * 0.5)