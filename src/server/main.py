#!/usr/bin/env python3
"""
FastAPI server for HVAC Controller API
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
import dateutil.parser
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from concurrent.futures import ThreadPoolExecutor
import traceback
import asyncio
import argparse
import uvicorn

# Import our modules
from src.utils.config import (
    ControllerConfig, BuildingConfig, VentilationConfig, CO2SourceConfig, 
    get_controller_config, get_building_config, Config, FullConfig, config
)
from src.utils.factory import create_building_model, create_co2_sources
from src.controllers.hvac import HvacController
from src.controllers.ventilation.models import (
    WindowVentilationModel, ERVModel, NaturalVentilationModel, CO2Source,
    RoomCO2Dynamics
)
from src.models.weather import WeatherConditions, SolarIrradiation
from src.utils.timeseries import TimeSeries


# Pydantic models for API requests/responses
class WeatherData(BaseModel):
    """Weather data for a single time point"""
    hour: float
    outdoor_temperature: float
    wind_speed: float = 5.0
    solar_altitude_rad: float = 0.5
    solar_azimuth_rad: float = 0.0
    solar_intensity_w: float = 800.0
    ground_temperature: float = 12.0


class WeatherTimeSeries(BaseModel):
    """Weather data for a single time point"""
    time: str
    outdoor_temperature: float
    wind_speed: float = 5.0
    solar_altitude_rad: float = 0.5
    solar_azimuth_rad: float = 0.0
    solar_intensity_w: float = 800.0
    ground_temperature: float = 12.0


class ControlRequest(BaseModel):
    """Request for control optimization"""
    current_co2_ppm: float
    current_temp_c: float
    current_time_hours: float
    weather_data: List[WeatherData]


class ControlResponse(BaseModel):
    """Response with control actions"""
    ventilation_controls: List[float]
    hvac_control: float
    total_cost: float
    predicted_co2_ppm: float
    predicted_temp_c: float


class PredictionRequest(BaseModel):
    """Request for getting next prediction"""
    current_co2_ppm: float
    current_temp_c: float
    current_time: str = Field(description="Current time in ISO format (e.g., '2024-01-15T09:30:00Z')")
    weather_time_series: List[WeatherTimeSeries]
    horizon_hours: Optional[float] = Field(default=None, description="Custom horizon in hours (uses controller default if not specified)")


class PredictionResponse(BaseModel):
    """Response with prediction data"""
    next_prediction: Optional[List[float]]
    time_horizon_hours: List[float]
    weather_forecast: List[dict]
    has_prediction: bool
    co2_trajectory: Optional[List[float]] = Field(default=None, description="Predicted CO2 levels over time horizon")
    temperature_trajectory: Optional[List[float]] = Field(default=None, description="Predicted temperature levels over time horizon")


class ModelInfo(BaseModel):
    """Information about the building and controller models"""
    building_components: List[str]
    heating_system_type: str
    ventilation_types: List[str]
    controller_params: dict


# Global controller instance
controller: Optional[HvacController] = None
building_model = None
room_dynamics = None
last_weather_series: Optional[TimeSeries] = None # Added global variable for weather series


def create_ventilation_models(building_config):
    """Create ventilation models from building configuration"""
    # Create window ventilation models
    window_ventilations = []
    for window_config in building_config.ventilation.window_ventilations:
        window_vent = WindowVentilationModel(
            max_airflow_m3_per_hour=window_config.max_airflow_m3_per_hour
        )
        window_ventilations.append(window_vent)
    
    # Create ERV models
    erv_ventilations = []
    for erv_config in building_config.ventilation.ervs:
        erv_vent = ERVModel(
            heat_recovery_efficiency=erv_config.heat_recovery_efficiency,
            fan_power_w_m3_per_hour=erv_config.fan_power_w_m3_per_hour,
            max_airflow_m3_per_hour=erv_config.max_airflow_m3_per_hour
        )
        erv_ventilations.append(erv_vent)
    
    # Create natural ventilation models
    natural_ventilations = []
    for natural_config in building_config.ventilation.natural_ventilations:
        natural_vent = NaturalVentilationModel(
            indoor_volume_m3=natural_config.indoor_volume_m3,
            infiltration_rate_ach=natural_config.infiltration_rate_ach
        )
        natural_ventilations.append(natural_vent)
    
    controllable_ventilations = window_ventilations + erv_ventilations
    return controllable_ventilations, natural_ventilations


def initialize_controller():
    """Initialize the HVAC controller with current configuration"""
    global controller, building_model, room_dynamics
    
    # Get configuration
    controller_config = get_controller_config()
    building_config = get_building_config()
    full_config = config.full_config
    
    # Create building model from configuration
    building_model = create_building_model(building_config)
    
    # Create ventilation models from config
    controllable_ventilations, natural_ventilations = create_ventilation_models(building_config)
    
    # Create CO2 sources from config
    co2_sources = create_co2_sources(building_config.room.co2_sources)
    
    # Create room dynamics
    room_dynamics = RoomCO2Dynamics(
        volume_m3=building_config.room.volume_m3,
        sources=co2_sources,
        controllable_ventilations=controllable_ventilations,
        natural_ventilations=natural_ventilations,
        outdoor_co2_ppm=building_config.room.outdoor_co2_ppm
    )
    
    # TODO: inject schedlue into controller instead of setting later
    controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=controller_config.horizon_hours,
        co2_weight=controller_config.co2_weight,
        energy_weight=controller_config.energy_weight,
        comfort_weight=controller_config.comfort_weight,
        step_size_hours=controller_config.step_size_hours,
        optimization_method=controller_config.optimization_method,
        max_iterations=controller_config.max_iterations,
    )
    
    # Load and translate weekly schedules if available
    print(f"DEBUG: full_config type: {type(full_config)}")
    print(f"DEBUG: full_config has schedules: {hasattr(full_config, 'schedules')}")
    print(f"DEBUG: full_config.schedules: {getattr(full_config, 'schedules', None)}")
    
    if full_config.schedules and full_config.schedules.get("weekly_schedule"):
        print(f"Found weekly schedule in config: {full_config.schedules['weekly_schedule'].keys()}")
        absolute_schedule = full_config.schedules.get("weekly_schedule")        
        # Translate weekly schedule to absolute times
        # Set the translated schedule in controller
        controller.set_saved_schedule(absolute_schedule)
    else:
        print("No weekly schedule found in config")
        print(f"DEBUG: full_config.schedules is: {full_config.schedules}")
        if hasattr(full_config, 'schedules') and full_config.schedules:
            print(f"DEBUG: full_config.schedules keys: {full_config.schedules.keys()}")
        raise ValueError("No schedule provided")
    # Create controller
    
    print(f"Initialized controller with building model and {len(controllable_ventilations)} ventilation types")


def weather_data_to_timeseries(weather_data: List[WeatherData]) -> TimeSeries:
    """Convert weather data to TimeSeries"""
    time_points = [wd.hour for wd in weather_data]
    weather_conditions = []
    for wd in weather_data:
        solar = SolarIrradiation(
            altitude_rad=wd.solar_altitude_rad,
            azimuth_rad=wd.solar_azimuth_rad,
            intensity_w=wd.solar_intensity_w
        )
        
        weather = WeatherConditions(
            irradiation=solar,
            wind_speed=wd.wind_speed,
            outdoor_temperature=wd.outdoor_temperature,
            ground_temperature=wd.ground_temperature
        )
        weather_conditions.append(weather)
    return TimeSeries(time_points, weather_conditions)

def weather_time_series_to_relative_time(weather_time_series: List[WeatherTimeSeries], start_time: datetime) -> TimeSeries:
    try:
        time_points = [(dateutil.parser.isoparse(x.time) - start_time).total_seconds() / 3600.0 for x in weather_time_series]
    except TypeError as e:
        print(e, 'time series datetime', weather_time_series[0].time, 'start time', start_time)
    weather_conditions = []
    for point in weather_time_series:
        solar = SolarIrradiation(
            altitude_rad=point.solar_altitude_rad,
            azimuth_rad=point.solar_azimuth_rad,
            intensity_w=point.solar_intensity_w
        )
        
        weather = WeatherConditions(
            irradiation=solar,
            wind_speed=point.wind_speed,
            outdoor_temperature=point.outdoor_temperature,
            ground_temperature=point.ground_temperature
        )
        weather_conditions.append(weather)
    return TimeSeries(time_points, weather_conditions)

# FastAPI app
app = FastAPI(
    title="HVAC Controller API",
    description="API for MPC-based HVAC control with CO2 and temperature management",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize controller on startup"""
    initialize_controller()


@app.get("/")
async def root(request: Request):
    """Root endpoint with HTML dashboard"""
    return Jinja2Templates(directory="templates").TemplateResponse(
        "index.html", {"request": request}
    )

@app.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        # Return the current config without reloading
        config.load_config("hvac_config.json")
        return config.full_config.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")

@app.post("/config")
async def save_config_endpoint(config_data: dict):
    """Save configuration"""
    try:
        # Validate the config data
        full_config = FullConfig(**config_data)
        
        # Save to file
        config.save_to_file(full_config, "hvac_config.json")
        
        # Reinitialize controller with new config
        global controller
        initialize_controller()
        
        return {"message": "Configuration saved and controller reinitialized successfully"}
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid configuration")


@app.post("/control")
async def get_control(request: ControlRequest):
    """Get optimal control actions for current state"""
    if controller is None:
        raise HTTPException(status_code=500, detail="Controller not initialized")
    
    try:
        # Convert weather data to TimeSeries
        weather_series = weather_data_to_timeseries(request.weather_data)
        
        # Get controls
        ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
            request.current_co2_ppm,
            request.current_temp_c,
            weather_series,
            request.current_time_hours
        )
        
        # Get predictions (simplified)
        predicted_co2 = request.current_co2_ppm + 50  # Placeholder
        predicted_temp = request.current_temp_c + 0.1  # Placeholder
        
        return ControlResponse(
            ventilation_controls=ventilation_controls,
            hvac_control=hvac_controls[0],
            total_cost=total_cost,
            predicted_co2_ppm=predicted_co2,
            predicted_temp_c=predicted_temp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Control optimization failed:")


@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    """Get next prediction from controller"""

    if controller is None:
        raise HTTPException(status_code=500, detail="Controller not initialized")
    
    try:
        # print("Running predict with request", request)
        # Convert current_time to hours since start of day
        current_time = datetime.now()
        if request.current_time:
            # Use provided time of day directly
            print(request.current_time)
            current_time = dateutil.parser.isoparse(request.current_time)
            print(dateutil.parser.isoparse(request.current_time))
        # Convert weather data to TimeSeries
        weather_series = weather_time_series_to_relative_time(request.weather_time_series, current_time)
        # Use custom horizon if provided
        print("About to set horizon")
        if request.horizon_hours is not None and request.horizon_hours != controller.horizon_hours:
            controller.set_horizon(request.horizon_hours)
        
        # Store weather series for plotting
        # Note: We'll store this in a global variable since HvacController doesn't have this attribute
        global last_weather_series
        last_weather_series = weather_series
        # Run optimization in a thread pool to avoid blocking
        print("About to run predict")
            # Run the optimization in a separate thread

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Run the optimization in a separate thread
            ventilation_controls, hvac_controls, total_cost = await loop.run_in_executor(
                executor,
                controller.optimize_controls,
                request.current_co2_ppm,
                request.current_temp_c,
                weather_series,
                current_time
            )
        
        # Get next prediction array
        next_prediction = controller.get_next_prediction()
        
        # Get predicted trajectories for CO2 and temperature
        co2_trajectory = None
        temperature_trajectory = None
        
        if next_prediction is not None:
            # Extract control sequences from the prediction
            n_ventilation_vars = controller.n_ventilation * controller.n_steps
            ventilation_vector = next_prediction[:n_ventilation_vars]
            hvac_vector = next_prediction[n_ventilation_vars:]
            
            # Reshape into sequences
            ventilation_sequences = []
            for i in range(controller.n_ventilation):
                start_idx = i * controller.n_steps
                end_idx = (i + 1) * controller.n_steps
                ventilation_sequences.append(ventilation_vector[start_idx:end_idx].tolist())
            
            hvac_sequence = hvac_vector.tolist()
            
            # Predict trajectories
            co2_trajectory, temperature_trajectory = controller.predict_trajectories(
                request.current_co2_ppm,
                request.current_temp_c,
                [ventilation_sequences, [hvac_sequence]],
                weather_series,
                current_time
            )
        
        # Create time horizon
        time_horizon = [i * controller.step_size_hours 
                       for i in range(controller.n_steps)]
        
        # Extract weather forecast for the horizon
        weather_forecast = []
        for t in time_horizon:
            if t <= weather_series.ticks[-1]:
                weather = weather_series.interpolate(t)
                weather_forecast.append({
                    "hour": t,
                    "outdoor_temperature": weather.outdoor_temperature,
                    "wind_speed": weather.wind_speed,
                    "solar_intensity_w": weather.irradiation.intensity
                })
            else:
                # Extrapolate with last known weather
                last_weather = weather_series.raw_values[-1]
                weather_forecast.append({
                    "hour": t,
                    "outdoor_temperature": last_weather.outdoor_temperature,
                    "wind_speed": last_weather.wind_speed,
                    "solar_intensity_w": last_weather.irradiation.intensity
                })
        
        return PredictionResponse(
            next_prediction=next_prediction.tolist() if next_prediction is not None else None,
            time_horizon_hours=time_horizon,
            weather_forecast=weather_forecast,
            has_prediction=next_prediction is not None,
            co2_trajectory=co2_trajectory,
            temperature_trajectory=temperature_trajectory
        )
        
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb_str)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {tb_str}")


@app.get("/plot-prediction")
async def plot_prediction():
    """Generate a plot of the last prediction from the controller"""
    if controller is None:
        print("[DEBUG] Controller is None!")
        raise HTTPException(status_code=500, detail="Controller not initialized")
    
    try:
        return controller.generate_plot()
        
    except Exception as e:
        import traceback
        print("[DEBUG] Exception in plot_prediction:")
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb_str)
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {tb_str}")


@app.get("/model-info")
async def get_model_info():
    """Get complete configuration information about the building and controller models"""
    if building_model is None or room_dynamics is None:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    try:
        # Get the complete configuration
        controller_config = get_controller_config()
        building_config = get_building_config()
        
        # Return the complete configuration
        return {
            "model_summary": {
                "building_components": [type(model).__name__ for model in building_model.thermal_models],
                "heating_system_type": type(building_model.heating_model).__name__,
                "ventilation_types": [type(vent).__name__ for vent in room_dynamics.controllable_ventilations],
                "natural_ventilation_types": [type(vent).__name__ for vent in room_dynamics.natural_ventilations],
                "co2_sources": [type(source).__name__ for source in room_dynamics.sources]
            },
            "controller_config": controller_config.model_dump(),
            "building_config": building_config.model_dump(),
        }
        
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb_str)
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {tb_str}")


def main():
    """Run the server"""
    parser = argparse.ArgumentParser(description="HVAC Controller API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config-file", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration if specified
    if args.config_file:
        config.load_config(args.config_file)
    
    # Run server
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main() 