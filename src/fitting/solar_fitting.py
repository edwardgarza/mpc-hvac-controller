import pandas as pd
import pvlib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.special import expit  # sigmoid
from math import sin, cos, radians


def encode_angles(alt_deg, az_deg):
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    return np.column_stack([np.sin(az), np.cos(az), np.sin(alt)])

def day_factor(alt_deg, kappa=50.0):
    return expit(kappa * np.sin(np.deg2rad(alt_deg)))  # smooth near horizon


def train_solar_GPR_kernel(df_train):
    # --- prepare training arrays (assume aligned pandas DataFrame `df_train`)
    # df_train columns: 'DNI','alt_deg','az_deg','q_solar_into_house' (W/m^2)
    mask = (df_train['dni'] > 5.0) & (df_train['elevation'] > 0)
    X_ang = encode_angles(df_train.loc[mask,'elevation'].values,
                        df_train.loc[mask,'azimuth'].values)
    # y_perW = df_train.loc[mask,'q_solar_into_house'].values / df_train.loc[mask,'DNI'].values

    # apply log1p transform for stability & positivity
    y_train = np.log1p(np.maximum(0.0, y_perW))

    # kernel: constant * RBF + white noise
    kernel = ConstantKernel(1.0, (1e-3,1e3)) * RBF(length_scale=[1.0,1.0,0.5], length_scale_bounds=(1e-2,1e2)) \
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1.0))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=5)
    gp.fit(X_ang, y_train)
    return gp
    # --- prediction on new points (df_pred with DNI, alt_deg, az_deg)
    X_pred = encode_angles(df_pred['alt_deg'].values, df_pred['az_deg'].values)
    log1p_geom_pred, geom_std = gp.predict(X_pred, return_std=True)
    geom_pred = np.expm1(log1p_geom_pred)           # >= 0

    # enforce horizon gating
    df_pred['day_factor'] = day_factor(df_pred['alt_deg'].values, kappa=50.0)

    # final fdirect
    df_pred['fdirect'] = df_pred['DNI'].values * geom_pred * df_pred['day_factor'].values
    # clamp tiny negatives (numerical) to zero
    df_pred['fdirect'] = np.clip(df_pred['fdirect'], 0.0, None)



def estimate_irradiance_components(lat, long, cloud_cover_series):

    location = pvlib.location.Location(lat, long)
    times = cloud_cover_series.index

    # Step 1: Calculate clear-sky irradiance
    clearsky = location.get_clearsky(times, model='ineichen')

    # Corrected linear scaling with an offset 
    offset = 0.35
    cloudy_ghi = (offset + (1 - offset) * (1 - cloud_cover_series)) * clearsky['ghi']

    # Step 3: Decompose the cloudy GHI using the DISC model to get DNI
    solar_position = location.get_solarposition(times=times)
    print(solar_position)
    disc_irradiance = pvlib.irradiance.disc(
        ghi=cloudy_ghi,
        solar_zenith=solar_position['zenith'],
        datetime_or_doy=times
    )

    dni = disc_irradiance['dni']

    # Step 4: Calculate DHI using the physical relationship
    zenith_rad = np.deg2rad(solar_position['zenith'])
    dhi = cloudy_ghi - dni * np.cos(zenith_rad)

    # Set negative values to 0
    dhi[dhi < 0] = 0

    # Combine all results into a single DataFrame
    results = pd.DataFrame({
        'dni': dni,
        'dhi': dhi,
        'elevation': np.deg2rad(solar_position['elevation']),
        'azimuth': np.deg2rad(solar_position['azimuth']),
    })

    return results

if __name__ == "__main__":
    lat, long = 47.7, -122.1  
    tz = 'America/Los_Angeles'
    times = pd.date_range('2025-09-09 00:00', '2025-09-09 14:00', freq='h', tz=tz)

    # cloud_cover_series = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0], index=times)
    cloud_cover_series = pd.Series([0.1] * len(times), index=times)

    print(estimate_irradiance_components(lat, long, cloud_cover_series))







# Define location and time
# latitude, longitude = 47.7, -122.1  # Cottage Lake, WA
# tz = 'America/Los_Angeles'
# location = pvlib.location.Location(latitude, longitude, tz=tz)
# times = pd.date_range('2025-09-09 10:00', '2025-09-09 14:00', freq='h', tz=tz)

# # Step 1: Calculate clear-sky irradiance
# clearsky = location.get_clearsky(times, model='ineichen')

# # Step 2: Get cloud cover data and apply corrected scaling
# # Cloud cover data (0.0 = clear, 1.0 = fully overcast)
# cloud_cover_fraction = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0], index=times)

# # Corrected linear scaling with an offset (e.g., 0.35)
# offset = 0.35
# cloudy_ghi = (offset + (1 - offset) * (1 - cloud_cover_fraction)) * clearsky['ghi']

# # Step 3: Decompose the cloudy GHI using the DISC model to get DNI
# solar_position = location.get_solarposition(times=times)

# disc_irradiance = pvlib.irradiance.disc(
#     ghi=cloudy_ghi,
#     solar_zenith=solar_position['zenith'],
#     datetime_or_doy=times
# )

# dni = disc_irradiance['dni']

# # Step 4: Calculate DHI using the physical relationship
# zenith_rad = np.deg2rad(solar_position['zenith'])
# dhi = cloudy_ghi - dni * np.cos(zenith_rad)

# # Set negative values to 0
# dhi[dhi < 0] = 0

# # Combine all results into a single DataFrame
# results = pd.DataFrame({
#     'dni': dni,
#     'dhi': dhi
# })

# print(results)
