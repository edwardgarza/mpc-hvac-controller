import math
from abc import ABC, abstractmethod

import WeatherConditions


class ThermalTransfer(ABC):

    @abstractmethod
    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        pass

