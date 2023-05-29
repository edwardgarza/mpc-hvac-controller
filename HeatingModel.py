from abc import abstractmethod, ABC


class HeatingModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def energy_to_generate_heat(self, heat: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        pass


class ElectricResistanceHeatingModel(HeatingModel):

    def energy_to_generate_heat(self, heat: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        return heat


class HeatPumpHeatingModel(HeatingModel):

    def __int__(self, outdoor_offset: float = 3, indoor_offset: float = 10, hspf: float = 9):
        self.outdoor_offset = outdoor_offset
        self.indoor_offset = indoor_offset
        self.hspf = hspf

    def energy_to_generate_heat(self, heat: float, indoor_temperature: float, outdoor_temperature: float) -> float:
        # 6.09 * - 0.09delta t from research paper source. self.hspf/9.0 is the correction factor for
        # more/less efficient heat pumps
        delta_t = (indoor_temperature + self.indoor_offset) - (outdoor_temperature - self.outdoor_offset)
        cop = self.hspf / 9.0 * (6.09 - 0.09 * min(0.0, delta_t))
        return heat / cop
