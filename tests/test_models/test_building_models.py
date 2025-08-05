import unittest
from src.models.building import (Studs, WallModel)
from src.utils.orientation import Orientation
from src.models.weather import WeatherConditions, SolarIrradiation

class TestWallModel(unittest.TestCase):
    default_orientation = Orientation()
    indoor_temp = 22
    outdoor_weather = WeatherConditions(SolarIrradiation(0, 0, 0), 0, 30, 10)
    studs = Studs(0.038, 0.089, 0.406)

    def test_exterior_insulation(self):
        wall_no_insulation = WallModel(self.studs, 0, 1, self.default_orientation, exterior_insulation_r=0)
        wall_ext_insulation = WallModel(self.studs, 0, 1, self.default_orientation, exterior_insulation_r=2)
        no_ins_power = wall_no_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        ext_ins_power = wall_ext_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        
        # because it's hotter outside there should be heat transfer into the building
        self.assertGreater(ext_ins_power, 0)
        self.assertGreater(no_ins_power, ext_ins_power)

    def test_stud_bay_insulation_insulation(self):
        wall_no_insulation = WallModel(self.studs, 0, 1, self.default_orientation)
        wall_insulation = WallModel(self.studs, 2, 1, self.default_orientation)
        no_ins_power = wall_no_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        ins_power = wall_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        
        # because it's hotter outside there should be heat transfer into the building
        self.assertGreater(ins_power, 0)
        self.assertGreater(no_ins_power, ins_power)

    def test_stud_bay_vs_exterior_insulation(self):
        wall_ext_insulation = WallModel(self.studs, 0, 1, self.default_orientation, exterior_insulation_r=2)
        wall_bay_insulation = WallModel(self.studs, 2, 1, self.default_orientation)
        ext_ins_power = wall_ext_insulation.powerflow(self.indoor_temp, self.outdoor_weather)
        bay_ins_power = wall_bay_insulation.powerflow(self.indoor_temp, self.outdoor_weather)

        # because exterior insulation doesn't have thermal bridging it is more effective than stud bay insulation at the same R value
        self.assertGreater(bay_ins_power, ext_ins_power)