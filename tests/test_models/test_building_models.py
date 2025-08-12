import unittest
from src.models.building import (BuildingModel, RoofModel, Studs, WallModel)
from src.models.thermal_device import ElectricResistanceThermalDeviceModel, HeatPumpThermalDeviceModel
from src.utils.orientation import Orientation
from src.models.weather import WeatherConditions, SolarIrradiation

class TestWallModel(unittest.TestCase):
    default_orientation = Orientation()
    indoor_temp = 20
    outdoor_weather = WeatherConditions(SolarIrradiation(0, 0, 0), 0, 30, 10)
    studs = Studs(0.038, 0.089, 0.406)

    def test_exterior_insulation(self):
        wall_no_insulation = WallModel(self.studs, 0, 1, self.default_orientation, exterior_insulation_r=0)
        wall_ext_insulation = WallModel(self.studs, 0, 1, self.default_orientation, exterior_insulation_r=2)
        no_ins_power = wall_no_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        ext_ins_power = wall_ext_insulation.powerflow(self.indoor_temp, self.outdoor_weather)

        # because it's hotter outside there should be heat transfer into the building
        self.assertGreater(no_ins_power, ext_ins_power)

        # the power flowing should be slightly lower than 1 m^2 * 10K / r=2 or 5
        self.assertLess(ext_ins_power, 5)
        self.assertGreater(ext_ins_power, 4)
    def test_stud_bay_insulation_insulation(self):
        wall_no_insulation = WallModel(self.studs, 0, 1, self.default_orientation)
        wall_insulation = WallModel(self.studs, 2, 1, self.default_orientation)
        no_ins_power = wall_no_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        ins_power = wall_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        
        # because it's hotter outside there should be heat transfer into the building
        self.assertGreater(ins_power, 0)
        self.assertGreater(no_ins_power, ins_power)

        # power flow through the stud bay should be just over 1 m^2 * 10K / r=2 or 5 because of additional thermal bridging
        self.assertGreater(ins_power, 5)
        self.assertLess(ins_power, 6)

    def test_stud_bay_vs_exterior_insulation(self):
        wall_ext_insulation = WallModel(self.studs, 0, 1, self.default_orientation, exterior_insulation_r=2)
        wall_bay_insulation = WallModel(self.studs, 2, 1, self.default_orientation)
        ext_ins_power = wall_ext_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        bay_ins_power = wall_bay_insulation.powerflow(self.indoor_temp, self.outdoor_weather)

        # because exterior insulation doesn't have thermal bridging it is more effective than stud bay insulation at the same R value
        self.assertGreater(bay_ins_power, ext_ins_power)

    def test_roof_insulation(self):
        roof_model = RoofModel(10, 50, Orientation())
        roof_power = roof_model.powerflow(self.indoor_temp, self.outdoor_weather)
        
        # 50m^2 * 10K / 10R
        self.assertAlmostEqual(roof_power, 50, delta=0.1)
    
    def test_building_model(self):
        building_model = BuildingModel(
            [
                RoofModel(10, 50, Orientation()), 
                RoofModel(5, 50, Orientation())
            ], 
            HeatPumpThermalDeviceModel(), 
            10 ** 6
        )

        heat_flow = building_model.powerflow(self.indoor_temp, self.outdoor_weather)
        self.assertAlmostEqual(heat_flow, 150, delta=0.1)

        # no net heat flow

        hvac_input = building_model.heating_model.power_consumed(-150, self.indoor_temp, self.outdoor_weather.outdoor_temperature)
        temp_change_hvac = building_model.temperature_change_per_s(self.indoor_temp, self.outdoor_weather, -hvac_input, 0)
        temp_change_additional = building_model.temperature_change_per_s(self.indoor_temp, self.outdoor_weather, 0, -150)
        self.assertAlmostEqual(temp_change_hvac, 0)
        self.assertAlmostEqual(temp_change_additional, 0)

        temp_change_no_input = building_model.temperature_change_per_s(self.indoor_temp, self.outdoor_weather, 0, 0)
        self.assertAlmostEqual(temp_change_no_input, 150 / building_model.heat_capacity, delta=0.1)
