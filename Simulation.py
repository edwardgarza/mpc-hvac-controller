import tkinter
import matplotlib.pyplot
import numpy as np

from MpcController import ModelPredictiveControl
from BuildingModel import *
from HeatingModel import *
from WeatherConditions import WeatherConditions, SolarIrradiation

if __name__ == "__main__":
    wall = WallModel(Studs(1.5, 3.5, 16), 13, 100, None)
    window = WindowModel(4, 0, 0.7)
    roof = RoofModel(60, 0, None, 0.85)
    floor = PierAndBeam(Studs(1.5, 5.5, 16), 30, 0, None)
    heating_model = HeatPumpHeatingModel()
    # only the walls have non-zero area
    model = BuildingModel(wall, window, roof, floor, heating_model, 100)
    mpc = ModelPredictiveControl(model, 10, None, 0.01)
    inside_temperature = 40
    solar_irradiation = 0, 0, 0
    outside_weather = 0, 120, 0
    set_point = 100
    x0 = [inside_temperature, *solar_irradiation, *outside_weather, set_point]
    new_inside_temp = inside_temperature
    u_values = []

    inside_values = []
    for i in range(50):
        u0 = mpc.control(x0)
        new_inside_temp += model.temperature_change_per_s(
            new_inside_temp,
            WeatherConditions(SolarIrradiation(*solar_irradiation), *outside_weather),
            u0)
        u_values.append(u0)
        x0[0] = new_inside_temp
        inside_values.append(new_inside_temp)

    print(u_values)
    print(inside_values)
    matplotlib.pyplot.plot(u_values)
    matplotlib.pyplot.plot(inside_values)
    matplotlib.pyplot.show()