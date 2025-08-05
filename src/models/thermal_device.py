from abc import abstractmethod, ABC
from typing import Tuple


class ThermalDeviceModel(ABC):

    @abstractmethod
    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        pass

    @abstractmethod
    def power_consumed(self, output_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        pass

    @property   
    @abstractmethod
    def output_range(self) -> Tuple[float, float]:
        pass

class ElectricResistanceThermalDeviceModel(ThermalDeviceModel):

    def __init__(self, output_range: Tuple[float, float] = (0, 10000)):
        self._output_range = output_range

    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        return input_power
    
    def power_consumed(self, output_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        return output_power

    @property
    def output_range(self) -> Tuple[float, float]:
        return self._output_range


class HeatPumpThermalDeviceModel(ThermalDeviceModel):
    """
    Heat pump model that uses a simple model of the heat pump efficiency that has some temperature dependence.
    For an airconditioner simply make the max output range 0.
    """
    def __init__(self, outdoor_offset: float = 7, indoor_offset: float = 10, hspf: float = 9, output_range: Tuple[float, float] = (-10000, 10000)):
        self.outdoor_offset = outdoor_offset
        self.indoor_offset = indoor_offset
        self.hspf = hspf
        self._output_range = output_range

    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        # 6.09 * - 0.09delta_t from research paper source. self.hspf/9.0 is the correction factor for
        # more/less efficient heat pumps
        if input_power is None or input_power == 0:
            return 0
        heating = input_power > 0
        cop = self.cop(heating, indoor_temperature, outdoor_temperature)
        if heating:
            # for heating, output is cop * input
            return input_power * cop
        else:
            # approximate cooling as cop - 1
            return input_power * (cop - 1)
    
    def power_consumed(self, output_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        if output_power is None or output_power == 0:
            return 0
        heating = output_power > 0
        cop = self.cop(heating, indoor_temperature, outdoor_temperature)
        if heating:
            # for heating, output is cop * input
            return output_power / cop
        else:
            # approximate cooling as cop - 1
            return abs(output_power) / (cop - 1)

    def cop(self, heating: bool, indoor_temperature: float, outdoor_temperature: float) -> float:
        if heating:
            delta_t = (indoor_temperature + self.indoor_offset) - (outdoor_temperature - self.outdoor_offset)
        else:
            delta_t = (outdoor_temperature + self.outdoor_offset) - (indoor_temperature - self.indoor_offset)
        return self.hspf / 9.0 * (6.09 - 0.09 * max(0.0, delta_t))


    @property
    def output_range(self) -> Tuple[float, float]:
        return self._output_range