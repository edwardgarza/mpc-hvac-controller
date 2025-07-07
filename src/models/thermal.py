from abc import ABC, abstractmethod
from src.models.weather import WeatherConditions


class ThermalTransfer(ABC):

    @abstractmethod
    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        pass
