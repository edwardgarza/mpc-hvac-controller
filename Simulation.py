import matplotlib.pyplot as plt
import pandas as pd
from MpcController import ModelPredictiveControl
from BuildingModel import *
from HeatingModel import *
from TimeSeries import TimeSeries
from WeatherConditions import WeatherConditions, SolarIrradiation
from pandas import to_datetime

if __name__ == "__main__":
    wall = WallModel(Studs(1.5, 3.5, 16), 13, 100, None)
    window = WindowModel(4, 0, 0.7)
    roof = RoofModel(60, 0, None, 0.85)
    floor = PierAndBeam(Studs(1.5, 5.5, 16), 30, 0, None)
    heating_model = HeatPumpHeatingModel()
    # only the walls have non-zero area
    model = BuildingModel(wall, window, roof, floor, heating_model, 1000)
    mpc = ModelPredictiveControl(model, 20, None, 0.1)
    inside_temperature = 40
    outside_weather = [WeatherConditions(SolarIrradiation(0, 0, 0), 0, 30, 0) for x in range(55)]
    outside_weather_ts = TimeSeries([x for x in range(55)], outside_weather)

    set_point = 55
    new_inside_temp = inside_temperature
    u_values = []

    inside_values = []
    for i in range(20):
        u0 = mpc.control(inside_temperature, outside_weather[0].to_array(), set_point)
        new_inside_temp += model.temperature_change_per_s(
            new_inside_temp,
            outside_weather[i],
            u0)
        u_values.append(u0)
        inside_temperature = new_inside_temp
        inside_values.append(new_inside_temp)

    print([x.outdoor_temperature for x in outside_weather])
    print(u_values)
    print(inside_values)
    plt.plot([x.outdoor_temperature for x in outside_weather])
    plt.plot(u_values)
    plt.plot(inside_values)
    plt.show()
