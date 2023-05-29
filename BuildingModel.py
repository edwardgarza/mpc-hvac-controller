from abc import ABC

import Orientation
import WeatherConditions
from ThermalTransfer import ThermalTransfer


class Studs:

    def __init__(self, width, depth, spacing):
        self.width = width
        self.depth = depth
        self.spacing = spacing


class WallModel(ThermalTransfer):

    def __init__(self, studs: Studs, insulation_batts_r: float, area_sq_m: float, orientation: Orientation):
        self.studs = studs
        self.insulation_batts_r = insulation_batts_r
        self.area_sq_m = area_sq_m
        self.orientation = orientation

    def _rvalue(self) -> float:
        r_value_studs = self.studs.depth * 1
        # weighted sum of the r values
        return (self.studs.width * r_value_studs + (
                    self.studs.spacing - self.studs.width) * self.insulation_batts_r) / self.studs.spacing

    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        # guess at the color of the walls with 0.5 absorp. come back to this later
        return (inside_temperature - weather_conditions.sol_temp(0.5, self.orientation)) * self.area_sq_m / self._rvalue()


class RoofModel(ThermalTransfer):

    def __init__(self, insulation_r: float, area_sq_m: float, orientation: Orientation, absorptivity: float = 0.85):
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m
        self.absorptivity = absorptivity
        self.orientation = orientation

    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        return (inside_temperature - weather_conditions.sol_temp(self.absorptivity, self.orientation)) * \
            self.area_sq_m / self.insulation_r


class FloorModel(ABC, ThermalTransfer):
    pass


class SlabModel(FloorModel):

    def __init__(self, insulation_r: float, area_sq_m: float):
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m

    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        return (inside_temperature - weather_conditions.ground_temperature) * self.area_sq_m / self.insulation_r


class PierAndBeam(WallModel, FloorModel):
    """This is basically the same as a wall but with no irradiance."""

    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        return (inside_temperature - weather_conditions.outdoor_temperature) * self.area_sq_m / self.rvalue()


class WindowModel(ThermalTransfer):

    def __init__(self, insulation_r: float, area_sq_m: float, solar_heat_gain_coefficient):
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m
        self.shgc = solar_heat_gain_coefficient

    def powerflow(self, inside_temperature: float, weather_conditions: WeatherConditions):
        # thermal conduction and solar irradiance pass through. sol_temp is irrelevant for windows (?)
        return self.area_sq_m * ((inside_temperature - weather_conditions.outdoor_temperature) / self.insulation_r +
                                 self.shgc * weather_conditions.projected_intensity(True))


class BuildingModel(ThermalTransfer):

    def __init__(self, wall_model: WallModel,
                 window_model: WindowModel,
                 roof_model: RoofModel,
                 floor_model: FloorModel,
                 heat_capacity: float):
        self.thermal_models = {'wall': wall_model,
                               'window': window_model,
                               'roof': roof_model,
                               'floor': floor_model}  # type: dict[str, ThermalTransfer]
        self.heat_capacity = heat_capacity

    def powerflow(self, *args) -> float:
        # Heat flow out of the building. Negative indicates that heat is flowing in.
        return sum(map(lambda x, y: y.powerflow(args), self.thermal_models))

    def temperature_change_per_s(self, args) -> float:
        return self.powerflow(*args) / self.heat_capacity
