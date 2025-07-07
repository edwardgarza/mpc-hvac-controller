#!/usr/bin/env python3
"""
Integrated HVAC Controller combining heating/cooling and ventilation control
"""

import numpy as np
import scipy.optimize as optimize
from typing import List, Tuple, Optional, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from src.models.weather import WeatherConditions
from src.models.heating import HeatingModel
from src.models.building import BuildingModel
from src.controllers.ventilation.models import RoomCO2Dynamics, BaseVentilationModel
from src.utils.timeseries import TimeSeries


class HvacController:
    """
    Integrated HVAC controller that optimizes both heating/cooling and ventilation
    simultaneously to minimize total energy cost while maintaining comfort.
    
    Heating/cooling is controlled by a single variable:
    - Positive values: Heating power (kW)
    - Negative values: Cooling power (kW)
    """
    
    def __init__(self,
                 room_dynamics: RoomCO2Dynamics,
                 building_model: BuildingModel,
                 horizon_hours: float = 24,
                 co2_weight: float = 1.0,
                 energy_weight: float = 1.0,
                 comfort_weight: float = 1.0,
                 co2_target_ppm: float = 800,
                 temp_target_c: float = 22.0,
                 step_size_hours: float = 0.25,
                 optimization_method: str = "L-BFGS-B",
                 max_iterations: int = 500,
                 use_linear_trajectories: bool = True,
                 electricity_cost_per_kwh: float = 0.15):
        """
        Initialize integrated HVAC controller
        
        Args:
            room_dynamics: CO2 dynamics model
            building_model: Building thermal model (includes heating/cooling)
            horizon_hours: Prediction horizon in hours
            co2_weight: Weight for CO2 deviation penalty
            energy_weight: Weight for energy cost
            comfort_weight: Weight for temperature comfort
            co2_target_ppm: Target CO2 concentration
            temp_target_c: Target indoor temperature
            step_size_hours: Time step size
            optimization_method: Optimization method
            max_iterations: Maximum optimization iterations
        """
        self.room_dynamics = room_dynamics
        self.building_model = building_model
        self.horizon_hours = horizon_hours
        self.co2_weight = co2_weight
        self.energy_weight = energy_weight
        self.comfort_weight = comfort_weight
        self.co2_target_ppm = co2_target_ppm
        self.temp_target_c = temp_target_c
        self.step_size_hours = step_size_hours
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.use_linear_trajectories = use_linear_trajectories
        self.electricity_cost_per_kwh = electricity_cost_per_kwh
        # Time discretization
        self.step_size_seconds = step_size_hours * 3600
        self.n_steps = int(horizon_hours / step_size_hours)
        
        # Control dimensions
        self.n_ventilation = len(room_dynamics.controllable_ventilations)
        self.n_hvac = 1  # Single heating/cooling control
        self.n_controls = self.n_ventilation + self.n_hvac
        
        # Previous control for smoothing
        self.u_prev = None
        self.next_prediction = None
    
    def predict_trajectories(self, 
                           initial_co2_ppm: float,
                           initial_temp_c: float,
                           control_sequences: List[List[List[float]]],
                           weather_series_hours: TimeSeries,
                           start_time_hours: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Predict both CO2 and temperature trajectories
        
        Args:
            initial_co2_ppm: Starting CO2 concentration
            initial_temp_c: Starting temperature inside
            control_sequences: Control sequences [ventilation_controls, hvac_controls]
            weather_series_hours: TimeSeries of weather conditions
            start_time_hours: Starting time for prediction (hours from weather series start)
            
        Returns:
            Tuple of (co2_trajectory, temp_trajectory)
        """
        co2_trajectory = [initial_co2_ppm]
        temp_trajectory = [initial_temp_c]
        
        current_co2 = initial_co2_ppm
        current_temp = initial_temp_c
        
        ventilation_controls = control_sequences[0]
        hvac_controls = control_sequences[1]
        
        for i in range(self.n_steps):
            # Extract control inputs for this time step
            ventilation_inputs = [ventilation_controls[j][i] for j in range(self.n_ventilation)]
            hvac_input = hvac_controls[0][i]  # hvac_controls is a list of lists
            
            # Get weather at current prediction time
            current_time = start_time_hours + i * self.step_size_hours
            weather = weather_series_hours.interpolate(current_time)
            
            if self.use_linear_trajectories:
                current_co2 = self.room_dynamics.co2_change_per_s(current_co2, ventilation_inputs) * self.step_size_seconds + current_co2
            else:
                current_co2 = self.room_dynamics.co2_levels_in_t(current_co2, ventilation_inputs, self.step_size_seconds)
            
            # Predict temperature change using building model
            # Calculate ventilation heat load (additional to building model)
            ventilation_heat_load = 0.0
            for j, (vent_model, vent_input) in enumerate(
                zip(self.room_dynamics.controllable_ventilations, ventilation_inputs)
            ):
                heat_load = vent_model.energy_load_kw(
                    vent_input, current_temp, weather.outdoor_temperature
                )
                ventilation_heat_load += heat_load
            
            # Calculate natural ventilation heat load
            for natural_vent in self.room_dynamics.natural_ventilations:
                heat_load = natural_vent.energy_load_kw(
                    natural_vent.airflow_m3_per_hour(), current_temp, weather.outdoor_temperature
                )
                ventilation_heat_load += heat_load
            if self.use_linear_trajectories:
                # Use building model for temperature change (includes HVAC and thermal transfer)
                temp_change_per_s = self.building_model.temperature_change_per_s(
                    current_temp, weather, hvac_input, ventilation_heat_load * 1000
                )

                temp_change = temp_change_per_s * self.step_size_seconds
                # print(f"old temp: {current_temp} new temp: {current_temp + temp_change} Temp change: {temp_change} per s, {temp_change * self.step_size_hours} in {self.step_size_hours} hours")
                current_temp += temp_change
            else:
                raise ValueError("Non-Linear trajectories are not supported")
            # Store trajectories
            co2_trajectory.append(current_co2)
            temp_trajectory.append(current_temp)
        
        return co2_trajectory, temp_trajectory
    
    def comfort_cost(self, temperature_c: float) -> float:
        """
        Calculate comfort cost based on temperature deviation from target
        
        Args:
            temperature_c: Indoor temperature in Celsius
            
        Returns:
            Comfort cost (higher values indicate worse comfort)
        """
        deviation = abs(temperature_c - self.temp_target_c)
        return self.comfort_weight * deviation ** 2
        
    
    def co2_cost(self, co2_ppm: float) -> float:
        """
        Calculate CO2 cost
        
        Args:
            co2_ppm: CO2 concentration in ppm
            
        Returns:
            CO2 cost
        """
        if co2_ppm <= self.co2_target_ppm:
            return 0.0
        else:
            deviation = co2_ppm - self.co2_target_ppm
            return self.co2_weight * (deviation ** 2)
    
    def energy_cost(self, ventilation_inputs: List[float], 
                   hvac_input: float,
                   indoor_temp_c: float, 
                   outdoor_temp_c: float) -> float:
        """
        Calculate total energy cost for ventilation and HVAC. Note that this is not the same as the energy cost of the building model.

        Ventillation's only direct energy cost is the fan power, while hvac's costs is running the unit.
        
        Args:
            ventilation_inputs: Ventilation control inputs
            hvac_input: Heating/cooling control input (positive=heating, negative=cooling)
            indoor_temp_c: Indoor temperature
            outdoor_temp_c: Outdoor temperature
            
        Returns:
            Total energy cost per second
        """
        total_cost_per_s = 0.0
        
        # Ventilation energy cost
        for (vent_model, vent_input) in zip(self.room_dynamics.controllable_ventilations, ventilation_inputs):
            total_cost_per_s += vent_model.fan_power_w(vent_input) * self.electricity_cost_per_kwh / 3600 / 1000
                
        total_cost_per_s += abs(hvac_input) * self.electricity_cost_per_kwh / 3600 / 1000
        
        return total_cost_per_s
    
    def cost_function(self, control_vector: np.ndarray,
                     current_co2_ppm: float,
                     current_temp_c: float,
                     weather_series_hours: TimeSeries,
                     start_time_hours: float = 0.0) -> float:
        """
        Calculate total cost for integrated HVAC control
        
        Args:
            control_vector: Flattened array of [ventilation_controls, hvac_controls]
            current_co2_ppm: Current CO2 concentration
            current_temp_c: Current temperature
            weather_conditions: Weather conditions over horizon
            
        Returns:
            Total cost
        """
        # Reshape control vector
        n_ventilation_vars = self.n_ventilation * self.n_steps
        ventilation_vector = control_vector[:n_ventilation_vars]
        hvac_vector = control_vector[n_ventilation_vars:]
        
        # Reshape into sequences
        ventilation_sequences = []
        for i in range(self.n_ventilation):
            start_idx = i * self.n_steps
            end_idx = (i + 1) * self.n_steps
            ventilation_sequences.append(ventilation_vector[start_idx:end_idx].tolist())
        
        hvac_sequence = hvac_vector.tolist()
        
        # Predict trajectories
        co2_trajectory, temp_trajectory = self.predict_trajectories(
            current_co2_ppm, current_temp_c, 
            [ventilation_sequences, [hvac_sequence]], 
            weather_series_hours, start_time_hours
        )
        
        total_cost = 0.0
        
        for i in range(self.n_steps):
            # CO2 cost
            co2_cost = self.co2_cost(co2_trajectory[i]) * self.step_size_seconds
            
            # Comfort cost
            comfort_cost = self.comfort_cost(temp_trajectory[i])* self.step_size_seconds
            
            # Energy cost
            ventilation_inputs = [ventilation_sequences[j][i] for j in range(self.n_ventilation)]
            hvac_input = hvac_sequence[i]
            current_time = start_time_hours + i * self.step_size_hours
            weather = weather_series_hours.interpolate(current_time)
            outdoor_temp = weather.outdoor_temperature
            
            energy_cost = self.energy_cost(
                ventilation_inputs, hvac_input,
                temp_trajectory[i], outdoor_temp
            ) * self.step_size_seconds * self.energy_weight
            
            total_cost += co2_cost + comfort_cost + energy_cost
        
        # Control smoothing penalty
        if self.u_prev is not None:
            for i in range(self.n_ventilation):
                control_change = ventilation_sequences[i][0] - self.u_prev[i]
                total_cost += 0.0001 * (control_change ** 2)
            
            if len(self.u_prev) > self.n_ventilation:
                hvac_change = hvac_sequence[0] - self.u_prev[self.n_ventilation]
                total_cost += 0.001 * (hvac_change ** 2)
        
        return total_cost
    
    def optimize_controls(self,
                         current_co2_ppm: float,
                         current_temp_c: float,
                         weather_series_hours: TimeSeries,
                         start_time_hours: float = 0.0) -> Tuple[List[float], List[float], float]:
        """
        Optimize both ventilation and HVAC controls
        
        Args:
            current_co2_ppm: Current CO2 concentration
            current_temp_c: Current temperature
            weather_conditions: Weather conditions over horizon
            
        Returns:
            Tuple of (ventilation_controls, hvac_controls, total_cost)
        """
        # Ensure weather series has enough data for the horizon
        max_time_needed = start_time_hours + self.horizon_hours
        if max_time_needed > weather_series_hours.ticks[-1]:
            print(f"Warning: Weather forecast only goes to {weather_series_hours.ticks[-1]} hours, but need {max_time_needed} hours")
            # Use the last available weather for extrapolation
        
        # Initial guess
        if self.u_prev is not None:
            u0 = self.next_prediction
            # u0 = np.array(self.u_prev * self.n_steps)
        else:
            u0 = np.zeros(self.n_controls * self.n_steps)
        
        # Bounds
        bounds = []
        # Ventilation bounds
        for i in range(self.n_ventilation):
            max_rate = self.room_dynamics.controllable_ventilations[i].max_airflow_m3_per_hour
            bounds.extend([(0.0, max_rate) for _ in range(self.n_steps)])
        # HVAC bounds (heating and cooling)
        bounds.extend([(self.building_model.heating_model.output_range[0], self.building_model.heating_model.output_range[1]) for _ in range(self.n_steps)])
        
        # Optimize
        result = optimize.minimize(
            self.cost_function,
            u0,
            args=(current_co2_ppm, current_temp_c, weather_series_hours, start_time_hours),
            method=self.optimization_method,
            bounds=bounds,
            options={'maxiter': self.max_iterations}
        )

        if not result.success:
            print(f"Optimization failed: {result.message}")
        
        # Extract optimal controls or best guess if optimization fails
        n_ventilation_vars = self.n_ventilation * self.n_steps
        ventilation_vector = result.x[:n_ventilation_vars]
        hvac_vector = result.x[n_ventilation_vars:]
        ventilation_controls = []
        for i in range(self.n_ventilation):
            start_idx = i * self.n_steps
            ventilation_controls.append(ventilation_vector[start_idx])
        hvac_control = hvac_vector[0]
        total_cost = result.fun
        
        # Store for next iteration
        self.u_prev = ventilation_controls + [hvac_control]

        # remove the current time step and append something random to the current optimized control for the next initial guess for the optimization
        self.next_prediction = np.concatenate((result.x[self.n_ventilation + 1:], self.u_prev))
        return ventilation_controls, [hvac_control], total_cost

    def get_next_prediction(self) -> Optional[np.ndarray]:
        return self.next_prediction
    
    def get_control_info(self,
                        current_co2_ppm: float,
                        current_temp_c: float,
                        weather_series_hours: TimeSeries,
                        start_time_hours: float = 0.0) -> Tuple[List[float], List[float], List[float], List[float], float, Dict[str, Any]]:
        """
        Get detailed control information including predicted trajectories
        
        Args:
            current_co2_ppm: Current CO2 concentration
            current_temp_c: Current temperature
            weather_conditions: Weather conditions over horizon
            
        Returns:
            Tuple of (ventilation_controls, hvac_controls, co2_trajectory, temp_trajectory, total_cost, additional_info)
        """
        ventilation_controls, hvac_controls, total_cost = self.optimize_controls(
            current_co2_ppm, current_temp_c, weather_series_hours, start_time_hours
        )
        
        # Predict full trajectories
        ventilation_sequences = [[control] * self.n_steps for control in ventilation_controls]
        hvac_sequence = hvac_controls * self.n_steps
        
        co2_trajectory, temp_trajectory = self.predict_trajectories(
            current_co2_ppm, current_temp_c,
            [ventilation_sequences, [hvac_sequence]],
            weather_series_hours, start_time_hours
        )
        
        additional_info = {
            'final_co2': co2_trajectory[-1],
            'final_temp': temp_trajectory[-1],
            'max_co2': max(co2_trajectory),
            'min_temp': min(temp_trajectory),
            'max_temp': max(temp_trajectory),
            'hvac_mode': 'heating' if hvac_controls[0] >= 0 else 'cooling',
            'hvac_power': abs(hvac_controls[0])
        }
        
        return ventilation_controls, hvac_controls, co2_trajectory, temp_trajectory, total_cost, additional_info 