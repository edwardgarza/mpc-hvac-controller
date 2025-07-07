#!/usr/bin/env python3
"""
FastAPI server for HVAC Controller API
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
import asyncio
import argparse
import uvicorn

# Import our modules
from src.utils.config import ControllerConfig, BuildingConfig, VentilationConfig, CO2SourceConfig
from src.utils.factory import BuildingFactory
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


class SimulationRequest(BaseModel):
    """Request for simulation"""
    initial_co2_ppm: float = 800.0
    initial_temp_c: float = 22.0
    simulation_hours: float = 24.0
    weather_data: List[WeatherData]


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
    current_time_hours: float
    weather_data: List[WeatherData]


class PredictionResponse(BaseModel):
    """Response with prediction data"""
    next_prediction: Optional[List[float]]
    time_horizon_hours: List[float]
    weather_forecast: List[dict]
    has_prediction: bool


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


def create_co2_sources():
    """Create default CO2 sources"""
    occupant_source = CO2Source(co2_production_rate_m3_per_hour=0.02)  # 2 people
    return [occupant_source]


def initialize_controller():
    """Initialize the HVAC controller with current configuration"""
    global controller, building_model, room_dynamics
    
    # Get configuration
    controller_config = get_controller_config()
    building_config = get_building_config()
    
    # Create building model from configuration
    building_model = create_building_model(building_config)
    
    # Create ventilation models from config
    controllable_ventilations, natural_ventilations = create_ventilation_models(building_config)
    
    # Create CO2 sources from config
    from utils.factory import create_co2_sources
    co2_sources = create_co2_sources(building_config.room.co2_sources)
    
    # Create room dynamics
    room_dynamics = RoomCO2Dynamics(
        volume_m3=building_config.room.volume_m3,
        sources=co2_sources,
        controllable_ventilations=controllable_ventilations,
        natural_ventilations=natural_ventilations,
        outdoor_co2_ppm=building_config.room.outdoor_co2_ppm
    )
    
    # Create controller
    controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=controller_config.horizon_hours,
        co2_weight=controller_config.co2_weight,
        energy_weight=controller_config.energy_weight,
        comfort_weight=controller_config.comfort_weight,
        co2_target_ppm=controller_config.co2_target_ppm,
        temp_target_c=controller_config.temp_target_c,
        step_size_hours=controller_config.step_size_hours,
        optimization_method=controller_config.optimization_method,
        max_iterations=controller_config.max_iterations,
        use_linear_trajectories=True,
        electricity_cost_per_kwh=controller_config.electricity_cost_per_kwh
    )
    
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


# FastAPI app
app = FastAPI(
    title="HVAC Controller API",
    description="API for MPC-based HVAC control with CO2 and temperature management",
    version="1.0.0"
)

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
async def root():
    """Root endpoint with HTML dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HVAC Controller API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .form-group { margin: 10px 0; }
            label { display: inline-block; width: 150px; }
            input, textarea { width: 200px; padding: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { background: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 3px; white-space: pre-wrap; }
            .plot-container { text-align: center; margin: 20px 0; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>HVAC Controller API</h1>
            

            
            <div class="endpoint">
                <h3>Test Plot Prediction</h3>
                <button onclick="testPlot()" id="plot-button">Generate Plot</button>
                <div id="plot-result" class="plot-container"></div>
            </div>
            
            <div class="endpoint">
                <h3>API Endpoints</h3>
                <ul>
                    <li><strong>GET /config</strong> - Get current configuration</li>
                    <li><strong>POST /control</strong> - Get optimal control actions</li>
                    <li><strong>POST /predict</strong> - Get next prediction</li>
                    <li><strong>GET /plot-prediction</strong> - Generate prediction plot</li>
                    <li><strong>GET /model-info</strong> - Get model information</li>
                </ul>
            </div>
        </div>
        
        <script>
            function createWeatherData() {
                const weatherData = [];
                for (let hour = 0; hour <= 24; hour += 2) {
                    weatherData.push({
                        hour: hour,
                        outdoor_temperature: 15.0 + 5.0 * (hour / 24.0),
                        wind_speed: 5.0,
                        solar_altitude_rad: 0.5,
                        solar_azimuth_rad: 0.0,
                        solar_intensity_w: 800.0,
                        ground_temperature: 12.0
                    });
                }
                return weatherData;
            }
            
            async function testPlot() {
                const plotButton = document.getElementById('plot-button');
                const plotResult = document.getElementById('plot-result');
                
                // Show loading state
                plotButton.disabled = true;
                plotButton.textContent = 'Generating...';
                plotResult.textContent = 'Loading...';
                
                try {
                    // Generate plot directly (it will run prediction if needed)
                    const plotResponse = await fetch('/plot-prediction', {
                        method: 'GET'
                    });
                    
                    if (!plotResponse.ok) {
                        throw new Error(`Plot generation failed: ${plotResponse.status}`);
                    }
                    
                    const plotResultData = await plotResponse.json();
                    console.log('Plot response:', plotResultData);
                    if (plotResultData.plot_data) {
                        const img = document.createElement('img');
                        img.src = 'data:image/png;base64,' + plotResultData.plot_data;
                        img.alt = 'Prediction Plot';
                        plotResult.innerHTML = '';
                        plotResult.appendChild(img);
                        console.log('Plot image created and added to page');
                    } else {
                        plotResult.textContent = 'No plot data received';
                        console.log('No plot data in response');
                    }
                } catch (error) {
                    plotResult.textContent = 'Error: ' + error.message;
                } finally {
                    // Reset button state
                    plotButton.disabled = false;
                    plotButton.textContent = 'Generate Plot';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/config")
async def get_config():
    """Get current configuration"""
    controller_config = get_controller_config()
    building_config = get_building_config()
    
    return {
        "controller": controller_config.model_dump(),
        "building": building_config.model_dump()
    }


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
        raise HTTPException(status_code=500, detail=f"Control optimization failed: {str(e)}")


@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    """Get next prediction from controller"""
    if controller is None:
        raise HTTPException(status_code=500, detail="Controller not initialized")
    
    try:
        # Convert weather data to TimeSeries
        weather_series = weather_data_to_timeseries(request.weather_data)
        
        # Store weather series for plotting
        controller.last_weather_series = weather_series
        
        # Run optimization in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Run the optimization in a separate thread
            ventilation_controls, hvac_controls, total_cost = await loop.run_in_executor(
                executor,
                controller.optimize_controls,
                request.current_co2_ppm,
                request.current_temp_c,
                weather_series,
                request.current_time_hours
            )
        
        # Get next prediction array
        next_prediction = controller.get_next_prediction()
        
        # Create time horizon
        time_horizon = [request.current_time_hours + i * controller.step_size_hours 
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
                last_weather = weather_series.values[-1]
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
            has_prediction=next_prediction is not None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/plot-prediction")
async def plot_prediction():
    """Generate a plot of the last prediction from the controller"""
    if controller is None:
        raise HTTPException(status_code=500, detail="Controller not initialized")
    
    try:
        # Get the last prediction from the controller
        next_prediction = controller.get_next_prediction()
        
        if next_prediction is None:
            # No prediction available, run a default one
            from models.weather import WeatherConditions, SolarIrradiation
            
            # Create default weather data
            weather_data = []
            for hour in range(0, 25, 6):
                solar = SolarIrradiation(altitude_rad=0.5, azimuth_rad=0.0, intensity_w=800.0)
                weather = WeatherConditions(
                    irradiation=solar,
                    wind_speed=5.0,
                    outdoor_temperature=15.0 + 5.0 * (hour / 24.0),
                    ground_temperature=12.0
                )
                weather_data.append(weather)
            
            weather_series = TimeSeries([0, 6, 12, 18, 24], weather_data)
            controller.last_weather_series = weather_series
            
            # Run optimization with default parameters
            ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
                800.0,  # default CO2
                22.0,   # default temp
                weather_series,
                0.0     # default time
            )
            
            next_prediction = controller.get_next_prediction()
        
        # Create time horizon
        time_horizon = [i * controller.step_size_hours for i in range(controller.n_steps)]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Weather forecast (if we have weather data)
        if hasattr(controller, 'last_weather_series') and controller.last_weather_series is not None:
            weather_times = []
            weather_temps = []
            for t in time_horizon:
                if t <= controller.last_weather_series.ticks[-1]:
                    weather = controller.last_weather_series.interpolate(t)
                    weather_times.append(t)
                    weather_temps.append(weather.outdoor_temperature)
            
            if weather_times:
                ax1.plot(weather_times, weather_temps, 'b-', linewidth=2, label='Outdoor Temperature')
                ax1.set_ylabel('Temperature (°C)')
                ax1.set_title('Weather Forecast')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
        
        # Plot 2: Next prediction
        # Split prediction into ventilation and HVAC controls
        n_ventilation_vars = controller.n_ventilation * controller.n_steps
        ventilation_pred = next_prediction[:n_ventilation_vars]
        hvac_pred = next_prediction[n_ventilation_vars:]
        
        # Reshape ventilation predictions
        for i in range(controller.n_ventilation):
            start_idx = i * controller.n_steps
            end_idx = (i + 1) * controller.n_steps
            vent_sequence = ventilation_pred[start_idx:end_idx]
            ax2.plot(time_horizon, vent_sequence, 
                    label=f'Ventilation {i+1}', linewidth=2)
        
        # Plot HVAC control
        ax2.plot(time_horizon, hvac_pred, 'r-', linewidth=2, label='HVAC Control')
        ax2.set_ylabel('Control Value')
        ax2.set_xlabel('Time (hours)')
        ax2.set_title('Predicted Controls')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"plot_data": plot_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about the building and controller models"""
    if building_model is None or room_dynamics is None:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    # Building components
    building_components = []
    for model in building_model.thermal_models:
        building_components.append(type(model).__name__)
    
    # Heating system
    heating_system_type = type(building_model.heating_model).__name__
    
    # Ventilation types
    ventilation_types = []
    for vent in room_dynamics.controllable_ventilations:
        ventilation_types.append(type(vent).__name__)
    
    # Controller parameters
    controller_config = get_controller_config()
    controller_params = {
        "horizon_hours": controller_config.horizon_hours,
        "step_size_hours": controller_config.step_size_hours,
        "optimization_method": controller_config.optimization_method,
        "co2_target_ppm": controller_config.co2_target_ppm,
        "temp_target_c": controller_config.temp_target_c
    }
    
    return ModelInfo(
        building_components=building_components,
        heating_system_type=heating_system_type,
        ventilation_types=ventilation_types,
        controller_params=controller_params
    )


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
        from config import config
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