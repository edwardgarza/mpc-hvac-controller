import math

def absolute_humidity_from_relative(temperature_c: float, relative_humidity: float) -> float:
    '''Returns the aboslute humidity in g/m^3 from the temperature and relative humidity, expressed as a decimal between 0 and 1.0'''
    return 216.7 * relative_humidity * 6.112 / (273.15 + temperature_c) * math.exp(17.62 * temperature_c / (243.21 + temperature_c))


def relative_humidity_from_asbolute(temperature_c: float, absolute_humidity_g_m3):
    return absolute_humidity_g_m3 / (216.7 * 6.112 / (273.15 + temperature_c) * math.exp(17.62 * temperature_c / (243.21 + temperature_c)))


def absolute_humidity_change_per_s(temp1: float, rh1: float, temp2: float, rh2: float, volume_m3_1, airflow_m3_hr_into_1: float) -> float:
    airflow_m3_per_s = airflow_m3_hr_into_1 / 3600
    air_change_per_s = airflow_m3_per_s / volume_m3_1
    return air_change_per_s * (absolute_humidity_from_relative(temp2, rh2) - absolute_humidity_from_relative(temp1, rh1))