#!/usr/bin/env python3
"""
Integrated ventilation MPC example with multiple ventilation types
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co2_control.VentilationModels import (
    RoomCO2Dynamics, CO2Source, NaturalVentilationModel, 
    WindowVentilationModel, HRVModel, ERVModel
)
from co2_control.IntegratedVentilationMpcController import IntegratedVentilationMpcController
from WeatherConditions import WeatherConditions, SolarIrradiation


def create_weather_forecast(hours: int, n_steps: int, base_temp: float = 20.0) -> List[WeatherConditions]:
    """
    Generate a weather forecast for the given number of hours and steps.
    Returns a list of WeatherConditions, one per simulation step.
    """
    weather = []
    for i in range(n_steps):
        t = i / n_steps * hours  # Convert step index to time in hours
        outdoor_temp = base_temp + 15 * np.sin(2 * np.pi * t / 24)
        weather.append(WeatherConditions(SolarIrradiation(0, 0, 0), 2.0, outdoor_temp, 8.0))
    return weather


def run_integrated_ventilation_simulation():
    """
    Run a comprehensive simulation of integrated ventilation control
    """
    print("=== Integrated Ventilation MPC Simulation ===")
    print("Room setup: 2 occupants, natural ventilation, windows, HRV, and ERV")
    print()
    
    # Room parameters
    room_volume_m3 = 100.0
    occupant_count = 2
    
    # Create CO2 sources (occupants)
    co2_sources = [
        CO2Source(co2_production_rate_m3_per_hour=0.02)  # 0.02 m³/hour per person
    ]
    
    # Create ventilation models
    natural_ventilation = NaturalVentilationModel(
        indoor_volume_m3=room_volume_m3,
        infiltration_rate_ach=0.1  # 0.1 air changes per hour
    )
    
    window_ventilation = WindowVentilationModel()
    hrv_system = HRVModel(heat_recovery_efficiency=0.0, fan_power_w_m3_per_hour=10.5)
    erv_system = ERVModel(heat_recovery_efficiency=0.8, moisture_recovery_efficiency=0.5, fan_power_w_m3_per_hour=0.5)
    
    # Create room dynamics model
    room_dynamics = RoomCO2Dynamics(
        volume_m3=room_volume_m3,
        sources=co2_sources,
        controllable_ventilations=[window_ventilation, hrv_system, erv_system],
        natural_ventilations=[natural_ventilation],
        outdoor_co2_ppm=400
    )
    
    # Create MPC controller
    controller = IntegratedVentilationMpcController(
        room_dynamics=room_dynamics,
        horizon_hours=6.0,
        co2_weight=0.0005,
        energy_weight=5000.0,
        co2_target_ppm=800,
        co2_max_ppm=1200,
        step_size_hours=0.25,
        co2_cost_type="asymmetric_quadratic",
        control_smoothing_weight=0.004
    )
    
    # Simulation parameters
    simulation_hours = 24
    time_steps = int(simulation_hours / controller.step_size_hours)
    
    # Create weather forecast
    weather_forecast = create_weather_forecast(simulation_hours, time_steps, base_temp=10.0)
    
    # Ensure weather_forecast is long enough for all steps
    if len(weather_forecast) < time_steps:
        last_weather = weather_forecast[-1]
        weather_forecast.extend([last_weather] * (time_steps - len(weather_forecast)))
    
    # Initialize simulation state
    current_co2_ppm = 800.0 
    indoor_temp_c = 22.0
    
    # Storage for results
    time_hours = []
    co2_levels = []
    window_controls = []
    hrv_controls = []
    erv_controls = []
    total_ventilation_rates = []
    energy_costs = []
    co2_costs = []
    
    print(f"Simulation parameters:")
    print(f"  Duration: {simulation_hours} hours")
    print(f"  Time steps: {time_steps}")
    print(f"  Initial CO2: {current_co2_ppm} ppm")
    print(f"  Indoor temperature: {indoor_temp_c}°C")
    print(f"  Outdoor temperature range: {min(w.outdoor_temperature for w in weather_forecast[:time_steps]):.1f}°C to {max(w.outdoor_temperature for w in weather_forecast[:time_steps]):.1f}°C")
    print()
    
    # Run simulation
    for step in range(time_steps):
        current_time_hours = step * controller.step_size_hours
        
        # Get weather conditions for the prediction horizon
        weather_horizon = weather_forecast[step:step + controller.n_steps]
        
        # Get optimal control
        optimal_controls, predicted_trajectory, total_cost, additional_info = controller.get_control_info(
            current_co2_ppm, weather_horizon, indoor_temp_c
        )
        
        # Extract control inputs
        window_control = optimal_controls[0]
        hrv_control = optimal_controls[1]
        erv_control = optimal_controls[2]
        
        # Calculate actual ventilation rates
        window_rate = window_ventilation.airflow_m3_per_hour(window_control)
        hrv_rate = hrv_system.airflow_m3_per_hour(hrv_control)
        erv_rate = erv_system.airflow_m3_per_hour(erv_control)
        natural_rate = natural_ventilation.airflow_m3_per_hour()
        total_ventilation_rate = window_rate + hrv_rate + erv_rate + natural_rate
        
        # Calculate energy costs
        outdoor_temp = weather_horizon[0].outdoor_temperature
        window_energy = window_ventilation.energy_cost_per_s(window_rate, indoor_temp_c, outdoor_temp)
        hrv_energy = hrv_system.energy_cost_per_s(hrv_rate, indoor_temp_c, outdoor_temp)
        erv_energy = erv_system.energy_cost_per_s(erv_rate, indoor_temp_c, outdoor_temp)
        
        # Calculate natural ventilation energy cost
        natural_energy = 0.0
        for natural_ventilation in room_dynamics.natural_ventilations:
            natural_rate = natural_ventilation.airflow_m3_per_hour()
            natural_energy += natural_ventilation.energy_cost_per_s(natural_rate, indoor_temp_c, outdoor_temp)
        
        total_energy_cost_per_s = window_energy + hrv_energy + erv_energy + natural_energy
        
        # Calculate CO2 cost
        co2_cost = controller.co2_cost_function(current_co2_ppm)
        
        # Store results (use weighted costs for plotting to show what optimizer sees)
        time_hours.append(current_time_hours)
        co2_levels.append(current_co2_ppm)
        window_controls.append(window_control)
        hrv_controls.append(hrv_control)
        erv_controls.append(erv_control)
        total_ventilation_rates.append(total_ventilation_rate)
        energy_costs.append(total_energy_cost_per_s * controller.step_size_seconds * controller.energy_weight)
        co2_costs.append(co2_cost)  # Already includes co2_weight
        
        # Update CO2 level
        co2_change_per_s = room_dynamics.co2_change_per_s(current_co2_ppm, optimal_controls)
        co2_change = co2_change_per_s * controller.step_size_seconds
        current_co2_ppm += co2_change
        
        # Print progress every 16 steps
        if step % 16 == 0:
            print(f"Time: {current_time_hours:4.1f}h, CO2: {current_co2_ppm:6.1f} ppm, "
                  f"Window: {window_control:5.1f}, HRV: {hrv_control:5.1f}, ERV: {erv_control:5.1f}, "
                  f"Total: {total_ventilation_rate:6.1f} m³/h")
    
    # Print summary statistics
    print("\n=== Simulation Results ===")
    print(f"Final CO2 level: {current_co2_ppm:.1f} ppm")
    print(f"Average CO2 level: {np.mean(co2_levels):.1f} ppm")
    print(f"Max CO2 level: {np.max(co2_levels):.1f} ppm")
    print(f"Min CO2 level: {np.min(co2_levels):.1f} ppm")
    print(f"CO2 target violations: {sum(1 for co2 in co2_levels if co2 > controller.co2_target_ppm)}")
    print(f"Total energy cost: ${np.sum(energy_costs):.4f}")
    print(f"Average energy cost per hour: ${np.mean(energy_costs) * 4:.4f}")
    
    # Calculate natural ventilation cost for comparison
    # natural_ventilation_cost = 0.0
    # for step in range(time_steps):
    #     outdoor_temp = weather_forecast[step].outdoor_temperature
    #     for natural_ventilation in room_dynamics.natural_ventilations:
    #         natural_rate = natural_ventilation.airflow_m3_per_hour()
    #         natural_ventilation_cost += natural_ventilation.energy_cost_per_s(natural_rate, indoor_temp_c, outdoor_temp) * controller.step_size_seconds
    
    # print(f"Natural ventilation energy cost: ${natural_ventilation_cost:.4f}")
    # print(f"Controllable ventilation energy cost: ${np.sum(energy_costs) - natural_ventilation_cost:.4f}")
    # print(f"Total CO2 cost: {np.sum(co2_costs):.2f}")
    
    # Create plots
    create_plots(time_hours, co2_levels, window_controls, hrv_controls, erv_controls, 
                total_ventilation_rates, energy_costs, co2_costs, controller, weather_forecast, indoor_temps=[indoor_temp_c] * len(time_hours))
    
    return {
        'time_hours': time_hours,
        'co2_levels': co2_levels,
        'window_controls': window_controls,
        'hrv_controls': hrv_controls,
        'erv_controls': erv_controls,
        'total_ventilation_rates': total_ventilation_rates,
        'energy_costs': energy_costs,
        'co2_costs': co2_costs
    }


def create_plots(time_hours, co2_levels, window_controls, hrv_controls, erv_controls, 
                total_ventilation_rates, energy_costs, co2_costs, controller, weather_forecast=None, indoor_temps=None):
    """
    Create comprehensive plots of the simulation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Integrated Ventilation MPC Simulation Results', fontsize=16)
    
    # CO2 levels with temperatures
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()  # Create secondary y-axis for temperature
    
    # Plot CO2 levels on primary y-axis
    ax1.plot(time_hours, co2_levels, 'b-', linewidth=2, label='CO2 Level')
    ax1.axhline(y=controller.co2_target_ppm, color='g', linestyle='--', label='Target (800 ppm)')
    ax1.axhline(y=controller.co2_max_ppm, color='r', linestyle='--', label='Max (1200 ppm)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('CO2 Concentration (ppm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot temperatures on secondary y-axis
    if weather_forecast is not None:
        outdoor_temps = [weather.outdoor_temperature for weather in weather_forecast[:len(time_hours)]]
        if len(outdoor_temps) == len(time_hours):
            ax2.plot(time_hours, outdoor_temps, 'orange', linewidth=2, label='Outdoor Temp', linestyle='--')
    
    if indoor_temps is not None and len(indoor_temps) == len(time_hours):
        ax2.plot(time_hours, indoor_temps, 'black', linewidth=2, label='Indoor Temp', linestyle=':')
    
    ax2.set_ylabel('Temperature (°C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    
    ax1.set_title('Indoor CO2 Levels and Temperatures')
    
    # Ventilation controls with outdoor and indoor temperature
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()  # Create secondary y-axis for temperature
    
    # Plot ventilation controls on primary y-axis
    ax1.plot(time_hours, window_controls, 'g-', linewidth=2, label='Windows')
    ax1.plot(time_hours, hrv_controls, 'b-', linewidth=2, label='HRV')
    ax1.plot(time_hours, erv_controls, 'r-', linewidth=2, label='ERV')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Ventilation Rate (m³/hour)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot outdoor and indoor temperatures on secondary y-axis
    if weather_forecast is not None:
        outdoor_temps = [weather.outdoor_temperature for weather in weather_forecast[:len(time_hours)]]
        if len(outdoor_temps) == len(time_hours):
            ax2.plot(time_hours, outdoor_temps, 'orange', linewidth=2, label='Outdoor Temp', linestyle='--')
    
    if indoor_temps is not None and len(indoor_temps) == len(time_hours):
        ax2.plot(time_hours, indoor_temps, 'black', linewidth=2, label='Indoor Temp', linestyle=':')
    
    ax2.set_ylabel('Temperature (°C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    
    ax1.set_title('Ventilation Controls and Temperatures')
    
    # Total ventilation rate
    axes[1, 0].plot(time_hours, total_ventilation_rates, 'purple', linewidth=2, label='Total Ventilation')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Ventilation Rate (m³/hour)')
    axes[1, 0].set_title('Total Ventilation Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined cost analysis
    total_costs = np.array(energy_costs) + np.array(co2_costs)
    axes[1, 1].plot(time_hours, energy_costs, 'orange', linewidth=2, label='Energy Cost (Weighted)', alpha=0.7)
    axes[1, 1].plot(time_hours, co2_costs, 'red', linewidth=2, label='CO2 Cost (Weighted)', alpha=0.7)
    axes[1, 1].plot(time_hours, total_costs, 'black', linewidth=2, label='Total Cost (Weighted)')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Cost (Weighted)')
    axes[1, 1].set_title('Cost Breakdown (Weighted Costs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('co2_control/integrated_ventilation_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_ventilation_strategies():
    """
    Compare different ventilation strategies
    """
    print("\n=== Ventilation Strategy Comparison ===")
    
    # Room setup
    room_volume_m3 = 100.0
    co2_sources = [CO2Source(co2_production_rate_m3_per_hour=0.02)]
    natural_ventilation = NaturalVentilationModel(room_volume_m3, infiltration_rate_ach=0.1)
    
    # Create different ventilation configurations
    strategies = {
        'Windows Only': {
            'controllable': [WindowVentilationModel()],
            'max_rates': [150.0]
        },
        'HRV Only': {
            'controllable': [HRVModel(heat_recovery_efficiency=0.7)],
            'max_rates': [100.0]
        },
        'ERV Only': {
            'controllable': [ERVModel(heat_recovery_efficiency=0.8, moisture_recovery_efficiency=0.5)],
            'max_rates': [100.0]
        },
        'Windows + HRV': {
            'controllable': [WindowVentilationModel(), HRVModel(heat_recovery_efficiency=0.7)],
            'max_rates': [100.0, 80.0]
        },
        'Windows + HRV + ERV': {
            'controllable': [WindowVentilationModel(), HRVModel(heat_recovery_efficiency=0.7), ERVModel(heat_recovery_efficiency=0.8, moisture_recovery_efficiency=0.5)],
            'max_rates': [80.0, 60.0, 60.0]
        }
    }
    
    # Weather conditions (cold day)
    weather_conditions = [WeatherConditions(SolarIrradiation(0, 0, 0), 2.0, 5.0, 3.0)] * 16  # 4 hours at 5°C
    indoor_temp_c = 22.0
    initial_co2_ppm = 600.0
    
    results = {}
    
    for strategy_name, config in strategies.items():
        print(f"\nTesting {strategy_name}...")
        
        # Create room dynamics
        room_dynamics = RoomCO2Dynamics(
            volume_m3=room_volume_m3,
            sources=co2_sources,
            controllable_ventilations=config['controllable'],
            natural_ventilations=[natural_ventilation],
            outdoor_co2_ppm=400
        )
        
        # Create controller
        controller = IntegratedVentilationMpcController(
            room_dynamics=room_dynamics,
            horizon_hours=4.0,
            co2_weight=1.0,
            energy_weight=5.0,
            co2_target_ppm=800,
            co2_max_ppm=1200,
            step_size_hours=0.25
        )
        
        # Run simulation for 4 hours
        current_co2 = initial_co2_ppm
        total_energy_cost = 0.0
        total_co2_cost = 0.0
        total_natural_ventilation_cost = 0.0
        
        for step in range(16):  # 4 hours / 0.25 hours = 16 steps
            weather_horizon = weather_conditions[step:step + controller.n_steps]
            optimal_controls, _, total_cost, additional_info = controller.get_control_info(
                current_co2, weather_horizon, indoor_temp_c
            )
            
            # Update CO2
            co2_change_per_s = room_dynamics.co2_change_per_s(current_co2, optimal_controls)
            co2_change = co2_change_per_s * controller.step_size_seconds
            current_co2 += co2_change
            
            # Calculate total energy costs including natural ventilation
            outdoor_temp = weather_horizon[0].outdoor_temperature
            controllable_energy_cost = sum(additional_info['energy_costs_per_s']) * controller.step_size_seconds
            
            # Calculate natural ventilation energy cost
            natural_energy_cost = 0.0
            for natural_ventilation in room_dynamics.natural_ventilations:
                ventilation_rate = natural_ventilation.airflow_m3_per_hour()
                energy_cost_per_s = natural_ventilation.energy_cost_per_s(
                    ventilation_rate, indoor_temp_c, outdoor_temp
                )
                natural_energy_cost += energy_cost_per_s * controller.step_size_seconds
            
            total_energy_cost += controllable_energy_cost + natural_energy_cost
            total_natural_ventilation_cost += natural_energy_cost
            total_co2_cost += controller.co2_cost_function(current_co2)
        
        results[strategy_name] = {
            'final_co2': current_co2,
            'total_energy_cost': total_energy_cost,
            'total_co2_cost': total_co2_cost,
            'total_natural_ventilation_cost': total_natural_ventilation_cost,
            'total_cost': total_energy_cost + total_co2_cost
        }
        
        print(f"  Final CO2: {current_co2:.1f} ppm")
        print(f"  Energy cost: ${total_energy_cost:.4f}")
        print(f"  CO2 cost: {total_co2_cost:.2f}")
        print(f"  Natural ventilation cost: {total_natural_ventilation_cost:.2f}")
        print(f"  Total cost: {total_energy_cost + total_co2_cost:.2f}")
    
    # Print comparison summary
    print("\n=== Strategy Comparison Summary ===")
    print(f"{'Strategy':<20} {'Final CO2':<10} {'Energy Cost':<12} {'CO2 Cost':<10} {'Natural Ventilation Cost':<20} {'Total Cost':<10}")
    print("-" * 70)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:<20} {result['final_co2']:<10.1f} ${result['total_energy_cost']:<11.4f} "
              f"{result['total_co2_cost']:<10.2f} {result['total_natural_ventilation_cost']:<20.2f} {result['total_cost']:<10.2f}")


if __name__ == "__main__":
    # Run the main simulation
    results = run_integrated_ventilation_simulation()
    
    # Run strategy comparison
    compare_ventilation_strategies() 