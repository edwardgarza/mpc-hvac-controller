from abc import ABC, abstractmethod
from typing import List
from scipy.interpolate import interp1d
import numpy as np


class ArrayConversion(ABC):

    @classmethod
    @abstractmethod
    def from_array(cls, arr):
        pass

    @abstractmethod
    def to_array(self):
        pass


class TimeSeries:

    def __init__(self, ticks: List[float], values: List[ArrayConversion]):
        assert len(ticks) == len(values)
        self.ticks = ticks
        self.val_type = values[0]
        self.raw_values = values
        self.values = [x.to_array() for x in values]
        self.interp_func = interp1d(ticks, self.values, axis=0)

    def __getitem__(self, item):
        return self.raw_values[item]

    def interpolate(self, time: float):
        """Interpolate value at given time."""
        if time <= self.ticks[0]:
            return self.raw_values[0]
        elif time >= self.ticks[-1]:
            return self.raw_values[-1]
        
        val = self.interp_func(time)
        return self.val_type.from_array(val)

    def __len__(self):
        return len(self.ticks)
    
    def __repr__(self):
        return f"TimeSeries(ticks={self.ticks}, values={[x.to_array() for x in self.raw_values]})"


# Example usage with WeatherConditions
if __name__ == "__main__":
    from ..models.weather import WeatherConditions, SolarIrradiation
    
    # Example: 24-hour weather forecast
    hours = [0.0, 6.0, 12.0, 18.0, 24.0]
    
    # Create sample weather conditions
    weather_data = []
    for i, hour in enumerate(hours):
        # Simple example: temperature varies, other conditions constant
        temp = 15 + 5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
        solar = SolarIrradiation(
            altitude_rad=0.5 + 0.3 * np.sin(2 * np.pi * hour / 24),
            azimuth_rad=0.0,
            intensity_w=800 if 6 <= hour <= 18 else 0  # Day/night cycle
        )
        weather = WeatherConditions(
            irradiation=solar,
            wind_speed=5.0,
            outdoor_temperature=temp,
            ground_temperature=temp - 2
        )
        weather_data.append(weather)
    print(len(weather_data), len(hours))
    weather_series = TimeSeries(hours, weather_data)
    
    print(weather_series)    # Example usage
    weather_5_7: WeatherConditions = weather_series.interpolate(5.7)
    weather_6_3: WeatherConditions = weather_series.interpolate(6.3)
    print(f"Weather at 5.7 hours: {weather_5_7.outdoor_temperature:.1f}°C")
    print(f"Weather at 6.3 hours: {weather_6_3.outdoor_temperature:.1f}°C")
    
    # Get subset for daytime hours
