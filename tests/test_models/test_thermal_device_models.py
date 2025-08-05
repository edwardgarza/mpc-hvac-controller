import unittest
from src.models.thermal_device import (ElectricResistanceThermalDeviceModel, HeatPumpThermalDeviceModel)


class TestElectricResistanceThermalDeviceModel(unittest.TestCase):

    def test_electric_resistance_output_equals_input(self):
        model = ElectricResistanceThermalDeviceModel((0, 1000))
        target_power_output = 100
        self.assertAlmostEqual(model.power_consumed(target_power_output, 0, 0), target_power_output)


class TestHeatPumpThermalDeviceModel(unittest.TestCase):
    input_power = 100
    input_power_cooling = -100
    indoor_temp = 22
    outdoor_temp = 10
    outdoor_temp_cooling = 30

    def test_heat_pump_produced_consumed_consistent(self):
        model = HeatPumpThermalDeviceModel(outdoor_offset=10, indoor_offset=10, hspf=10, output_range=(-10000, 10000))
        output_power = model.power_produced(self.input_power, self.indoor_temp, self.outdoor_temp)
        self.assertGreater(output_power, self.input_power)
        self.assertAlmostEqual(model.power_consumed(output_power, self.indoor_temp, self.outdoor_temp), self.input_power)

    def test_heat_pump_efficiency_temp_offset(self):
        # running a smaller delta t should result in a more efficient energy transfer
        less_efficient_model = HeatPumpThermalDeviceModel(outdoor_offset=10, indoor_offset=10, hspf=10, output_range=(-10000, 10000))
        more_efficient_model = HeatPumpThermalDeviceModel(outdoor_offset=5, indoor_offset=5, hspf=10, output_range=(-10000, 10000))
        self.assertLess(less_efficient_model.power_produced(self.input_power, self.indoor_temp, self.outdoor_temp), 
                        more_efficient_model.power_produced(self.input_power, self.indoor_temp, self.outdoor_temp))
        self.assertGreater( less_efficient_model.power_produced(self.input_power_cooling, self.indoor_temp, self.outdoor_temp_cooling), 
                            more_efficient_model.power_produced(self.input_power_cooling, self.indoor_temp, self.outdoor_temp_cooling))
    
    def test_heat_pump_efficiency_hspf(self):
        less_efficient_model = HeatPumpThermalDeviceModel(hspf=10, output_range=(-10000, 10000))
        more_efficient_model = HeatPumpThermalDeviceModel(hspf=12, output_range=(-10000, 10000))
        self.assertLess(less_efficient_model.power_produced(self.input_power, self.indoor_temp, self.outdoor_temp), 
                        more_efficient_model.power_produced(self.input_power, self.indoor_temp, self.outdoor_temp))
        self.assertGreater( less_efficient_model.power_produced(self.input_power_cooling, self.indoor_temp, self.outdoor_temp_cooling), 
                            more_efficient_model.power_produced(self.input_power_cooling, self.indoor_temp, self.outdoor_temp_cooling))

