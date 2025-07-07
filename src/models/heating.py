from abc import abstractmethod, ABC
from typing import Tuple


class HeatingModel(ABC):

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

class ElectricResistanceHeatingModel(HeatingModel):

    def __init__(self, output_range: Tuple[float, float] = (0, 10000)):
        self._output_range = output_range

    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        return input_power
    
    def power_consumed(self, output_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        return output_power

    @property
    def output_range(self) -> Tuple[float, float]:
        return self._output_range


class HeatPumpHeatingModel(HeatingModel):
    """
    Heat pump model that uses a simple model of the heat pump efficiency that has some temperature dependence.
    For an airconditioner simply make the max output range 0.
    """
    def __init__(self, outdoor_offset: float = 3, indoor_offset: float = 5, hspf: float = 9, output_range: Tuple[float, float] = (-10000, 10000)):
        self.outdoor_offset = outdoor_offset
        self.indoor_offset = indoor_offset
        self.hspf = hspf
        self._output_range = output_range

    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        # 6.09 * - 0.09delta t from research paper source. self.hspf/9.0 is the correction factor for
        # more/less efficient heat pumps
        if input_power is None or input_power == 0:
            return 0
        heating = input_power > 0
        if heating:
            delta_t = (indoor_temperature + self.indoor_offset) - (outdoor_temperature - self.outdoor_offset)
        else:
            delta_t = (outdoor_temperature + self.outdoor_offset) - (indoor_temperature - self.indoor_offset)
        cop = self.hspf / 9.0 * (6.09 - 0.09 * min(0.0, delta_t))
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
        if heating:
            delta_t = (indoor_temperature + self.indoor_offset) - (outdoor_temperature - self.outdoor_offset)
        else:
            delta_t = (outdoor_temperature + self.outdoor_offset) - (indoor_temperature - self.indoor_offset)
        cop = self.hspf / 9.0 * (6.09 - 0.09 * min(0.0, delta_t))
        if heating:
            # for heating, output is cop * input
            return output_power / cop
        else:
            # approximate cooling as cop - 1
            return abs(output_power) / (cop - 1)

    @property
    def output_range(self) -> Tuple[float, float]:
        return self._output_range