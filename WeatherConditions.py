import math

import Orientation


class SolarIrradiation:

    def __init__(self, altitude_rad: float, azimuth_rad: float, intensity_w: float):
        self.altitude = altitude_rad
        self.azimuth = azimuth_rad
        self.intensity = intensity_w

    def horizontal_intensity(self) -> float:
        return math.fabs(math.cos(math.pi / 2 - self.altitude)) * self.intensity

    def projected_intensity(self, orientation: Orientation) -> float:
        # calculate the dot product of the sun and the orientation
        return 0


class WeatherConditions:

    def __init__(self, irradiation: SolarIrradiation, wind_speed: float, outdoor_temperature: float, ground_temperature: float):
        self.irradiation = irradiation
        self.wind_speed = wind_speed
        self.outdoor_temperature = outdoor_temperature
        self.ground_temperature = ground_temperature

    def sol_temp(self, absorptivity: float, orientation: Orientation) -> float:
        return self.outdoor_temperature + absorptivity * self.irradiation.projected_intensity(orientation) / \
            (5.7 + 3.8 * self.wind_speed) - 3.9
