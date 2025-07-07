#!/usr/bin/env python3
"""
Example demonstrating the integrated HVAC controller with TimeSeries weather data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.controllers.ventilation.models import (
    RoomCO2Dynamics, WindowVentilationModel, HRVModel, 
    ERVModel, NaturalVentilationModel, CO2Source
)
from src.models.building import BuildingModel, WallModel, WindowModel, RoofModel, PierAndBeam, Studs
from src.models.heating import HeatPumpHeatingModel, ElectricResistanceHeatingModel
from src.controllers.hvac import HvacController
from src.models.weather import WeatherConditions, SolarIrradiation
from src.utils.orientation import Orientation
from src.utils.timeseries import TimeSeries


def create_example_room():
    """Create a simple room setup for testing"""
    
    # Create ventilation models
    window_vent = WindowVentilationModel()
    erv_vent = ERVModel(heat_recovery_efficiency=0.9, fan_power_w_m3_per_hour=0.3)
    natural_vent = NaturalVentilationModel(indoor_volume_m3=100.0, infiltration_rate_ach=0.2)
    
    # Create CO2 sources (occupants)
    occupant_source = CO2Source(co2_production_rate_m3_per_hour=0.02)  # 2 people
    
    # Create room dynamics
    room_dynamics = RoomCO2Dynamics(
        volume_m3=100.0,
        sources=[occupant_source],
        controllable_ventilations=[window_vent, erv_vent],
        natural_ventilations=[natural_vent],
        outdoor_co2_ppm=400
    )
    
    return room_dynamics


def create_example_building():
    """Create a simple building model"""
    
    # Create building components
    wall = WallModel(Studs(1.5, 3.5, 16), 13, 100, Orientation())
    window = WindowModel(4, 20, 0.7)  # 20 m² of windows
    roof = RoofModel(60, 50, Orientation(), 0.85)  # 50 m² roof
    floor = PierAndBeam(Studs(1.5, 5.5, 16), 30, 50, Orientation())  # 50 m² floor
    
    # Create heating model
    heating_model = HeatPumpHeatingModel(hspf=9.0, output_range=(-10000, 10000))
    # heating_model = ElectricResistanceHeatingModel()
    # Create building model
    building_model = BuildingModel(
        thermal_models=[wall, window, roof, floor],
        heating_model=heating_model,
        heat_capacity=10 ** 6  # J/K
    )
    
    return building_model


def create_weather_timeseries(forecast_hours=48):
    """Create a TimeSeries of weather conditions for the forecast period."""
    
    # Create time points (every 3 hours for forecast data)
    time_points = [float(x) for x in range(0, forecast_hours + 1)]
    weather_conditions = []
    
    for hour in time_points:
        # Simple sinusoidal temperature variation
        outdoor_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)
        
        # Create weather conditions
        solar = SolarIrradiation(
            altitude_rad=0.5,  # Simple fixed values
            azimuth_rad=0.0,
            intensity_w=800 * max(0, np.sin(2 * np.pi * hour / 24))
        )
        
        weather = WeatherConditions(
            irradiation=solar,
            wind_speed=5.0,
            outdoor_temperature=outdoor_temp,
            ground_temperature=12.0
        )
        
        weather_conditions.append(weather)
    
    # Create TimeSeries
    weather_series = TimeSeries(time_points, weather_conditions)
    return weather_series


def run_hvac_example():
    """Run the integrated HVAC controller example with TimeSeries weather data"""
    
    print("Setting up integrated HVAC controller example with TimeSeries weather...")
    
    # Create models
    room_dynamics = create_example_room()
    building_model = create_example_building()
    energy_cost_per_kwh = 0.15
    
    # Create controller
    controller = HvacController(
        room_dynamics=room_dynamics,
        building_model=building_model,
        horizon_hours=6.0,
        co2_weight=0.1,
        energy_weight=3000.0,
        comfort_weight=0.001,
        co2_target_ppm=800,
        temp_target_c=22.0,
        step_size_hours=0.5,
        optimization_method="SLSQP",
        max_iterations=500,
        use_linear_trajectories=True,
        electricity_cost_per_kwh=energy_cost_per_kwh
    )
    
    # Initial conditions
    current_co2_ppm = 2000.0
    current_temp_c = 17.1

    print(f"Initial conditions: CO2={current_co2_ppm} ppm, Temp={current_temp_c}°C")
    print(f"Targets: CO2={controller.co2_target_ppm} ppm, Temp={controller.temp_target_c}°C")
    
    # Create weather TimeSeries
    weather_series = create_weather_timeseries(48)  # 48-hour forecast
    print(f"Weather forecast: {len(weather_series)} points from 0 to {weather_series.ticks[-1]} hours")
    
    # Simulation parameters
    simulation_hours = 24  # Run for 24 hours
    simulation_steps = int(simulation_hours / controller.step_size_hours)
    
    # Storage for results
    co2_history = [current_co2_ppm]
    temp_history = [current_temp_c]
    ventilation_history = []
    hvac_history = []
    cost_history = []
    
    # Cost breakdown tracking
    co2_cost_history = []
    energy_cost_history = []
    comfort_cost_history = []
    
    print(f"\nRunning step-by-step simulation for {simulation_steps} steps ({simulation_hours} hours)...")
    
    for step in range(simulation_steps):
        # Current simulation time
        current_time = step * controller.step_size_hours
        
        # Get control actions for next step using TimeSeries
        ventilation_controls, hvac_controls, total_cost = controller.optimize_controls(
            current_co2_ppm, current_temp_c, weather_series, current_time
        )
        
        # Debug: Print detailed cost breakdown
        if step % 4 == 0:
            print(f"\n=== Step {step} (Time: {current_time:.1f}h) Cost Analysis ===")
            print(f"Ventilation controls: {ventilation_controls}")
            print(f"HVAC control: {hvac_controls[0]:.2f} kW")
            print(f"Total cost: {total_cost:.3f}")
            
            # Get current weather for display
            current_weather = weather_series.interpolate(current_time)
            if current_weather is not None:
                print(f"Current weather: {current_weather.outdoor_temperature:.1f}°C")
            else:
                print(f"Current weather: interpolated at {current_time:.1f}h")
            
            # Calculate individual costs for current state
            co2_cost = controller.co2_cost(current_co2_ppm)
            comfort_cost = controller.comfort_cost(current_temp_c)
            
            # Calculate energy cost for current controls
            total_energy_cost = 0.0
            for j, (vent_model, vent_input) in enumerate(
                zip(room_dynamics.controllable_ventilations, ventilation_controls)
            ):
                fan_power = vent_model.fan_power_w(vent_input)
                fan_cost = fan_power * energy_cost_per_kwh / 3600 / 1000  # $/s
                total_energy_cost += fan_cost
                print(f"  Vent {j} fan power: {fan_power:.4f} kW, cost: {fan_cost:.6f} $/s")
            
            hvac_cost = abs(hvac_controls[0]) * energy_cost_per_kwh / 3600
            total_energy_cost += hvac_cost
            print(f"  HVAC cost: {hvac_cost:.6f} $/s")
            print(f"  Total energy cost: {total_energy_cost:.6f} $/s")
            print(f"  CO2 cost: {co2_cost:.6f}")
            print(f"  Comfort cost: {comfort_cost:.6f}")
            print(f"  Weighted energy cost: {total_energy_cost * controller.step_size_seconds * controller.energy_weight:.3f}")
            print(f"  Weighted comfort cost: {comfort_cost * controller.step_size_seconds * controller.comfort_weight:.3f}")
            print("=" * 50)
        
        # Store control actions
        ventilation_history.append(ventilation_controls.copy())
        hvac_history.append(hvac_controls[0])
        cost_history.append(total_cost)
        
        # Calculate cost breakdown for current state
        co2_cost = controller.co2_cost(current_co2_ppm)
        comfort_cost = controller.comfort_cost(current_temp_c)
        
        # Calculate energy cost for current controls
        total_energy_cost = 0.0
        
        # Ventilation fan power cost only (not heat load)
        for j, (vent_model, vent_input) in enumerate(
            zip(room_dynamics.controllable_ventilations, ventilation_controls)
        ):
            vent_rate = vent_model.airflow_m3_per_hour(vent_input)
            fan_power_w = vent_model.fan_power_w(vent_rate)
            # Only fan power costs energy, not the heat load
            fan_cost_per_s = fan_power_w * energy_cost_per_kwh / 3600 / 1000
            total_energy_cost += fan_cost_per_s
        
        # Natural ventilation has no fan power, so no energy cost
        # (heat load is handled by HVAC/comfort costs)
        
        # HVAC energy cost
        hvac_input = hvac_controls[0]
        if hvac_input != 0:
            hvac_cost_per_s = abs(hvac_input) * energy_cost_per_kwh / 3600 / 1000
            total_energy_cost += hvac_cost_per_s
        # Store cost breakdown
        co2_cost_history.append(co2_cost)
        energy_cost_history.append(total_energy_cost * controller.step_size_seconds * controller.energy_weight)
        comfort_cost_history.append(comfort_cost)
        
        # Apply first step of controls
        ventilation_inputs = ventilation_controls
        hvac_input = hvac_controls[0]
        
        # Step forward in time
        # CO2 change
        new_co2_ppm = room_dynamics.co2_levels_in_t(current_co2_ppm, ventilation_inputs, controller.step_size_seconds)
        print(f"New CO2: {new_co2_ppm} old: {current_co2_ppm}, ventilation: {ventilation_inputs}")
        current_co2_ppm = new_co2_ppm
        # Temperature change
        # Calculate ventilation heat load
        ventilation_heat_load = 0.0
        for j, (vent_model, vent_input) in enumerate(
            zip(room_dynamics.controllable_ventilations, ventilation_inputs)
        ):
            heat_load = vent_model.energy_load_kw(
                vent_input, current_temp_c, current_weather.outdoor_temperature
            )
            ventilation_heat_load += heat_load
        # Calculate natural ventilation heat load
        for natural_vent in room_dynamics.natural_ventilations:
            heat_load = natural_vent.energy_load_kw(
                natural_vent.airflow_m3_per_hour(), current_temp_c, current_weather.outdoor_temperature
            )
            ventilation_heat_load += heat_load
        
        # Use building model for proper temperature integration
        new_current_temp_c = building_model.integrate_temperature_change(
            current_temp_c, current_weather, hvac_input, controller.step_size_seconds, ventilation_heat_load * 1000
        )
        print(f"New temp: {new_current_temp_c} old: {current_temp_c}, hvac: {hvac_input}, ventilation: {ventilation_heat_load}")
        current_temp_c = new_current_temp_c
        # Store state
        co2_history.append(current_co2_ppm)
        temp_history.append(current_temp_c)
        
        # Print progress every few steps
        if step % 4 == 0:
            print(f"Step {step}: CO2={current_co2_ppm:.1f} ppm, Temp={current_temp_c:.1f}°C, "
                  f"HVAC={hvac_input:.2f} W, Vent={ventilation_inputs}")
            print(f"  Costs: CO2={co2_cost:.3f}, Energy={energy_cost_history[-1]:.3f}, Comfort={comfort_cost:.3f}")
    
    # Print final results
    print("\n=== Simulation Results ===")
    print(f"Final CO2: {current_co2_ppm:.1f} ppm")
    print(f"Final temperature: {current_temp_c:.1f}°C")
    print(f"Max CO2: {max(co2_history):.1f} ppm")
    print(f"Temperature range: {min(temp_history):.1f}°C to {max(temp_history):.1f}°C")
    print(f"Average cost per step: {np.mean(cost_history):.3f}")
    
    # Plot results
    plot_simulation_results(co2_history, temp_history, ventilation_history, hvac_history, weather_series, 
                          co2_cost_history, energy_cost_history, comfort_cost_history, controller)
    
    return controller, room_dynamics, building_model


def plot_simulation_results(co2_history, temp_history, ventilation_history, hvac_history, weather_series, 
                          co2_cost_history, energy_cost_history, comfort_cost_history, controller):
    """Plot the step-by-step simulation results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time axis - use the actual step size from controller
    time_hours = np.arange(len(co2_history)) * controller.step_size_hours
    
    # CO2 trajectory
    ax1.plot(time_hours, co2_history, 'b-', linewidth=2, label='CO2 Level')
    ax1.axhline(y=800, color='r', linestyle='--', label='Target CO2')
    ax1.axhline(y=400, color='gray', linestyle=':', label='Outdoor CO2')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('CO2 (ppm)')
    ax1.set_title('CO2 Concentration Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Temperature trajectory - both indoor and outdoor
    ax2.plot(time_hours, temp_history, 'g-', linewidth=2, label='Indoor Temperature')
    
    # Extract outdoor temperatures from weather series
    outdoor_temps = [weather_series.interpolate(x).outdoor_temperature for x in time_hours]

    ax2.plot(time_hours, outdoor_temps, 'orange', linewidth=2, label='Outdoor Temperature', alpha=0.7)
    
    ax2.axhline(y=22, color='r', linestyle='--', label='Target Temperature')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Indoor vs Outdoor Temperature')
    ax2.legend()
    ax2.grid(True)
    
    # Cost breakdown over time
    cost_time_hours = np.arange(len(co2_cost_history)) * controller.step_size_hours
    
    # Add HVAC control on secondary y-axis of temperature plot
    ax2_twin = ax2.twinx()
    hvac_color = 'blue'
    hvac_label = 'Heating' if np.mean(hvac_history) >= 0 else 'Cooling'
    ax2_twin.plot(cost_time_hours, hvac_history, color=hvac_color, linewidth=1, label=hvac_label, alpha=0.7)
    ax2_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2_twin.set_ylabel('HVAC Power (W)', color=hvac_color)
    ax2_twin.legend(loc='upper right')
    ax3.plot(cost_time_hours, co2_cost_history, 'b-', linewidth=2, label='CO2 Cost')
    ax3.plot(cost_time_hours, energy_cost_history, 'r-', linewidth=2, label='Energy Cost')
    ax3.plot(cost_time_hours, comfort_cost_history, 'g-', linewidth=2, label='Comfort Cost')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Cost')
    ax3.set_title('Cost Breakdown Over Time')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')  # Log scale to see differences better
    
    # Ventilation controls over time
    vent_labels = ['Windows', 'ERV']
    vent_colors = ['blue', 'purple']
    
    # Extract ventilation sequences
    for i in range(2):  # Assuming 2 ventilation systems
        vent_sequence = [step[i] for step in ventilation_history]
        ax4.plot(cost_time_hours, vent_sequence, color=vent_colors[i], linewidth=2, label=vent_labels[i])
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Airflow (m³/h)')
    ax4.set_title('Ventilation Controls Over Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('hvac_controller_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSimulation results plotted and saved to 'hvac_controller_simulation.png'")
    
    # Print cost summary
    print(f"\n=== Cost Summary ===")
    print(f"Average CO2 cost: {np.mean(co2_cost_history):.3f}")
    print(f"Average Energy cost: {np.mean(energy_cost_history):.3f}")
    print(f"Average Comfort cost: {np.mean(comfort_cost_history):.3f}")
    print(f"Total average cost: {np.mean(co2_cost_history) + np.mean(energy_cost_history) + np.mean(comfort_cost_history):.3f}")


if __name__ == "__main__":
    run_hvac_example() 