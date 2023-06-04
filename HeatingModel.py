from abc import abstractmethod, ABC


class HeatingModel(ABC):

    @abstractmethod
    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        pass


class ElectricResistanceHeatingModel(HeatingModel):

    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        return 0 if input_power <= 0 else input_power


class HeatPumpHeatingModel(HeatingModel):

    def __init__(self, outdoor_offset: float = 3, indoor_offset: float = 5, hspf: float = 9):
        self.outdoor_offset = outdoor_offset
        self.indoor_offset = indoor_offset
        self.hspf = hspf

    def power_produced(self, input_power: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        # 6.09 * - 0.09delta t from research paper source. self.hspf/9.0 is the correction factor for
        # more/less efficient heat pumps
        if input_power is None or input_power == 0:
            return 0
        heating = input_power > 0
        if heating:
            delta_t = (indoor_temperature + self.indoor_offset) - (outdoor_temperature - self.outdoor_offset)
        else:
            delta_t = (indoor_temperature - self.indoor_offset) - (outdoor_temperature + self.outdoor_offset)
        cop = self.hspf / 9.0 * (6.09 - 0.09 * min(0.0, delta_t))
        if heating:
            # for heating, output is cop * input
            return input_power * cop
        else:
            # approximate cooling as cop - 1
            return input_power * (cop - 1)
