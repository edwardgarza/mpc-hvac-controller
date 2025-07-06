import math
from Orientation import Orientation
from TimeSeries import ArrayConversion


class SolarIrradiation(ArrayConversion):

    def __init__(self, altitude_rad: float, azimuth_rad: float, intensity_w: float):
        self.altitude = altitude_rad
        self.azimuth = azimuth_rad
        self.intensity = intensity_w

    @classmethod
    def from_array(cls, arr):
        return SolarIrradiation(*arr)

    def to_array(self):
        return [self.altitude, self.azimuth, self.intensity]

    def horizontal_intensity(self) -> float:
        return math.fabs(math.cos(math.pi / 2 - self.altitude)) * self.intensity


class WeatherConditions(ArrayConversion):

    def __init__(self, irradiation: SolarIrradiation, wind_speed: float, outdoor_temperature: float,
                 ground_temperature: float):
        self.irradiation = irradiation
        self.wind_speed = wind_speed
        self.outdoor_temperature = outdoor_temperature
        self.ground_temperature = ground_temperature

    @classmethod
    def from_array(cls, arr):
        solar_params = arr[0:3]
        remaining = arr[3:]
        return WeatherConditions(SolarIrradiation(*solar_params), *remaining)

    def to_array(self):
        return self.irradiation.to_array() + [self.wind_speed, self.outdoor_temperature, self.ground_temperature]

    def sol_temp(self, absorptivity: float, orientation: Orientation) -> float:
        sol_temp = self.outdoor_temperature + absorptivity * self.projected_intensity(orientation) / \
                   (5.7 + 3.8 * self.wind_speed)
        # sol_temp -= 3.9 * math.cos(vertical_orientation)
        return sol_temp

    def projected_intensity(self, orientation: Orientation) -> float:
        # calculate the dot product of the sun and the orientation
        return 0
