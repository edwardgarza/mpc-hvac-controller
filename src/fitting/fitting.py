import math
import pandas as pd
import requests
import scipy.optimize as optimize
import numpy as np
from scipy.optimize import curve_fit
import functools
from solar_fitting import estimate_irradiance_components

def to_celsius(value):
    # unavailable will be filtered out later
    if value == "unavailable":
        return value
    return (float(value) - 32) * 5/9


def query_home_assistant(token, sensors, start_time, end_time):
    url = f"http://homeassistant.local:8123/api/history/period/{start_time}"
    params = {
        "filter_entity_id": sensors,
        "end_time": end_time
    }
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers, params=params)
    return r.json()


def query_weather_data(token, weather_sensor, start_time, end_time):
    data = query_home_assistant(token, weather_sensor, start_time, end_time)

    rows = []
    for entity in data:
        for point in entity:
            if point['state'] in ["unavailable", '']:
                print("Skipping weather point due to bad data")
                continue
            rows.append({
                "entity_id": point["entity_id"],
                "time": point["last_changed"],
                "temperature": point['attributes']['temperature'] if False == '°C' else to_celsius(point['attributes']['temperature']),
                "humidity": point['attributes']['humidity'],
                "cloud_coverage": point['attributes']['cloud_coverage'],
            })
    df = pd.DataFrame(rows)
    val_cols = ["temperature", "humidity", "cloud_coverage"]
    for col in val_cols:
        df = df[df[col] != "unavailable"]                # filter bad values
        df[col] = pd.to_numeric(df[col], errors="coerce")  # make numeric
        df = df.dropna(subset=[col])                     # drop NaNs from conversion
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")              
    df = df.set_index("time", drop=True).sort_index()               
    return df


def query_homeassistant_temperatures(token, sensors, start_time, end_time):
    data = query_home_assistant(token, sensors, start_time, end_time)

    # Flatten into DataFrame
    rows = []
    for entity in data:
        for point in entity:
            rows.append({
                "entity_id": point["entity_id"],
                "time": point["last_changed"],
                "value": point["state"] if point['attributes']["unit_of_measurement"] == '°C' else to_celsius(point['state'])
            })
    df = pd.DataFrame(rows)

    # Convert types
    df = df[df["value"] != "unavailable"]                # filter bad values
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")              # convert time to datetime
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # make numeric
    df = df.dropna(subset=["value"])                     # drop NaNs from conversion
    df = df.set_index("time", drop=True).sort_index()               # use time as index
    return df


def thermal_model_no_sun(thermal_capacity, X, alpha, beta):
    Tout, Tin = X
    return (alpha + beta * (Tout - Tin)) / thermal_capacity


def thermal_model_diff_W(thermal_capacity, alpha, beta, X):
    Tout, Tin = X
    return (alpha + beta * (Tout - Tin)) / thermal_capacity


def run_temperature_no_hvac_no_sun_fit(token, s_in, s_out, start_time, end_time, thermal_capacity):
    '''Try to fit to building params when there is no hvac operating, ventilation active, or sun shining.'''

    df = query_homeassistant_temperatures(token, s_out + ',' + s_in, start_time, end_time)
    indoor = df[df["entity_id"] == s_in]["value"]
    outdoor = df[df["entity_id"] == s_out]["value"]

    freq = "5min"  
    indoor_rs = indoor.resample(freq).mean().interpolate(method="time")
    outdoor_rs = outdoor.resample(freq).mean().interpolate(method="time")

    # 2) Derivative on the resampled indoor series (per second)
    dt_sec = indoor_rs.index.to_series().diff().dt.total_seconds()
    dTdt = indoor_rs.diff() / dt_sec
    aligned = pd.concat([indoor_rs, outdoor_rs, dTdt], axis=1, keys=["Tin","Tout","dTdt"]).dropna()

    X = np.vstack([aligned["Tout"].values, aligned["Tin"].values])
    y = aligned["dTdt"].values

    # Fit parameters
    popt, pcov = curve_fit(functools.partial(thermal_model_no_sun, thermal_capacity), X, y)
    alpha, beta = popt

    print(f"Estimated α (in W): {start_time}", alpha)
    print(f"Estimated β (conduction in W/°C or R * m^2): {start_time}", beta)
    print(f"Estimated std_dev: {start_time}", np.sqrt(np.diag(pcov)))
    return popt, pcov


def run_temperature_no_hvac_sun_fit(token, s_in, s_out, s_weather, start_time, end_time, thermal_capacity, alpha, beta):
    '''Try to fit to irradiance transfer functions when there is no hvac operating, or ventilation active.
    
    For known alpha and beta, find the error between predicted and actual and attribute those fully to the 
    sun, and use that to train the irraiance transfer function.'''

    df = query_homeassistant_temperatures(token, s_out + ',' + s_in, start_time, end_time)
    df_weather = query_weather_data(token, s_weather, start_time, end_time)
    
    # example lat/long
    lat, long = 47, -122  
    print(estimate_irradiance_components(lat, long, df_weather["cloud_coverage"]))
    indoor = df[df["entity_id"] == s_in]["value"]
    outdoor = df[df["entity_id"] == s_out]["value"]

    freq = "5min"  
    indoor_rs = indoor.resample(freq).mean().interpolate(method="time")
    outdoor_rs = outdoor.resample(freq).mean().interpolate(method="time")
    # 2) Derivative on the resampled indoor series (per second)
    dt_sec = indoor_rs.index.to_series().diff().dt.total_seconds()
    dTdt = indoor_rs.diff() / dt_sec
    aligned = pd.concat([indoor_rs, outdoor_rs, dTdt], axis=1, keys=["Tin","Tout","dTdt"]).dropna()
    X = np.vstack([aligned["Tout"].values, aligned["Tin"].values])
    y = aligned["dTdt"].values
    power_diffs = (y - thermal_model_diff_W(thermal_capacity, alpha, beta, X)) * thermal_capacity # this is in units of J/K * K/s, which is W
    df_train = pd.Series(power_diffs, index=aligned.index)
    return df_train


if __name__ == "__main__":
    token = "HA_ACCESS_TOKEN"
    s_inside = "sensor.awair_element_66681_temperature"
    s_outside = "sensor.thermostat_outdoor_temperature"
    thermal_capacity_est = 10 ** 7 # thermal capacity is redundant with a and b so it needs to be separate
    vals = []
    alpha = 0
    beta = 0
    var_norm = 0

    # combine disjoint time slices by fitting to each separately and weighing by inverse variances
    # example time pairs - update for ones where no hvac was operating and there wasn't substantial solar gain
    for start_time, end_time in [("2025-09-08T08:00:00", "2025-09-08T14:00:00"), ("2025-09-09T08:00:00", "2025-09-09T14:00:00"), ("2025-09-10T01:00:00", "2025-09-10T10:00:00")]:
        fit_val = run_temperature_no_hvac_no_sun_fit(token, s_inside, s_outside, start_time, end_time, thermal_capacity_est)
        vals.append(fit_val)
        diag = np.diag(fit_val[1])
        alpha += fit_val[0][0] / diag[0]
        beta += fit_val[0][1] / diag[1]
        var_norm += 1 / diag
    print("\n")
    print("Estimated combined α (baseload heat in W):", alpha / var_norm[0], "std_dev", 1 / np.sqrt(var_norm[0]))
    print("Estimated combined β (conduction in W/°C or R * m^2): {start_time}", beta / var_norm[1], "std_dev", 1 / np.sqrt(var_norm[1]))
    start_time, end_time = ("2025-09-10T10:00:00", "2025-09-10T14:00:00")

    print('Solar fitting is still in progress - below is just for testing')
    # These power diffs are attributable to solar gain
    power_diffs = run_temperature_no_hvac_sun_fit(token, s_inside, s_outside, "weather.forecast_home", start_time, end_time, thermal_capacity_est, alpha, beta)
    print(power_diffs, np.mean(power_diffs))