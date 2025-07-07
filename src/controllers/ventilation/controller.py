#!/usr/bin/env python3
"""
Ventilation MPC Controller for CO2 control
"""

import numpy as np
import scipy.optimize as optimize
from typing import List, Tuple, Optional, Dict, Any
import time

from src.models.weather import WeatherConditions
from src.controllers.ventilation.models import RoomCO2Dynamics, BaseVentilationModel, CO2Source


class IntegratedVentilationMpcController:
    """
    Model Predictive Controller for integrated ventilation systems
    Balances CO2 levels with energy costs across multiple ventilation types
    """
    
    def __init__(self, 
                 room_dynamics: RoomCO2Dynamics,
                 horizon_hours: float = 24,
                 co2_weight: float = 1.0,
                 energy_weight: float = 1.0,
                 co2_target_ppm: float = 800,
                 co2_max_ppm: float = 1200,
                 step_size_hours: float = 0.25,
                 co2_cost_type: str = "asymmetric_quadratic",
                 control_smoothing_weight: float = 0.0001,
                 electricity_cost_per_kwh: float = 0.15,
                 use_parallel: bool = False,
                 max_iterations: int = 500,
                 optimization_method: str = "SLSQP",
                 use_analytical_gradients: bool = True):
        """
        Initialize integrated ventilation MPC controller
        
        Args:
            room_dynamics: RoomCO2Dynamics model for CO2 simulation
            horizon_hours: Prediction horizon in hours
            co2_weight: Weight for CO2 deviation from target
            energy_weight: Weight for energy costs
            co2_target_ppm: Target CO2 concentration
            co2_max_ppm: Maximum acceptable CO2 concentration
            step_size_hours: Time step size for optimization
            co2_cost_type: Type of CO2 cost function
            control_smoothing_weight: Weight for control change penalty
            electricity_cost_per_kwh: Electricity cost in dollars per kWh
            use_parallel: Whether to use parallel optimization
            max_iterations: Maximum number of iterations for optimization
            optimization_method: Optimization method to use ('SLSQP', 'trust-constr', 'L-BFGS-B', 'PMP')
            use_analytical_gradients: Whether to use analytical gradients (faster convergence)
        """
        self.room_dynamics = room_dynamics
        self.horizon_hours = horizon_hours
        self.co2_weight = co2_weight
        self.energy_weight = energy_weight
        self.co2_target_ppm = co2_target_ppm
        self.co2_max_ppm = co2_max_ppm
        self.step_size_hours = step_size_hours
        self.co2_cost_type = co2_cost_type
        self.control_smoothing_weight = control_smoothing_weight
        self.electricity_cost_per_kwh = electricity_cost_per_kwh
        self.use_parallel = use_parallel
        self.max_iterations = max_iterations
        self.optimization_method = optimization_method
        self.use_analytical_gradients = use_analytical_gradients
        
        # Convert time step to seconds
        self.step_size_seconds = step_size_hours * 3600
        
        # Number of control steps in horizon
        self.n_steps = int(horizon_hours / step_size_hours)
        
        # Number of controllable ventilation systems
        self.n_controllable = len(room_dynamics.controllable_ventilations)
        
        
        self.u_prev = None
        
    def predict_co2_trajectory(self, 
                             initial_co2_ppm: float,
                             control_sequences: List[List[float]],
                             weather_conditions: List[WeatherConditions]) -> List[float]:
        """
        Predict CO2 trajectory over the horizon
        
        Args:
            initial_co2_ppm: Starting CO2 concentration
            control_sequences: List of control sequences for each controllable ventilation type
            weather_conditions: Weather conditions for each time step
            
        Returns:
            Predicted CO2 concentrations for each time step
        """
        co2_trajectory = [initial_co2_ppm]
        current_co2 = initial_co2_ppm
        
        for i in range(self.n_steps):
            # Extract control inputs for this time step
            control_inputs = [control_sequences[j][i] for j in range(self.n_controllable)]
            
            # Calculate CO2 change for this time step
            co2_change_per_s = self.room_dynamics.co2_change_per_s(
                current_co2, 
                control_inputs
            )
            
            # Update CO2 concentration
            co2_change = co2_change_per_s * self.step_size_seconds
            current_co2 += co2_change
            co2_trajectory.append(current_co2)
        
        return co2_trajectory
    
    def co2_cost_function(self, co2_level: float) -> float:
        """
        Calculate CO2 cost using the selected cost function type
        
        Args:
            co2_level: Current CO2 concentration in ppm
            
        Returns:
            CO2 cost (higher values indicate worse air quality)
        """
        if self.co2_cost_type == "asymmetric_quadratic":
            return self._asymmetric_quadratic_cost(co2_level)
        elif self.co2_cost_type == "exponential":
            return self._exponential_cost(co2_level)
        elif self.co2_cost_type == "piecewise_linear":
            return self._piecewise_linear_cost(co2_level)
        elif self.co2_cost_type == "sigmoid":
            return self._sigmoid_cost(co2_level)
        else:
            raise ValueError(f"Unknown CO2 cost type: {self.co2_cost_type}")
    
    def _asymmetric_quadratic_cost(self, co2_level: float) -> float:
        """Asymmetric quadratic cost - no penalty below target, quadratic above"""
        if co2_level <= self.co2_target_ppm:
            return 0.0
        else:
            deviation = co2_level - self.co2_target_ppm
            return self.co2_weight * (deviation ** 2)
    
    def _exponential_cost(self, co2_level: float) -> float:
        """Exponential cost - aggressive penalty for high CO2"""
        if co2_level <= self.co2_target_ppm:
            return 0.0
        else:
            deviation = co2_level - self.co2_target_ppm
            return self.co2_weight * (np.exp(deviation / 200) - 1)
    
    def _piecewise_linear_cost(self, co2_level: float) -> float:
        """Piecewise linear cost with deadband around target"""
        deadband = 50  # ppm tolerance around target
        if co2_level <= self.co2_target_ppm + deadband:
            return 0.0
        elif co2_level <= self.co2_max_ppm:
            deviation = co2_level - (self.co2_target_ppm + deadband)
            return self.co2_weight * deviation
        else:
            # Heavy penalty for exceeding max
            return self.co2_weight * 1000 * (co2_level - self.co2_max_ppm)
    
    def _sigmoid_cost(self, co2_level: float) -> float:
        """Sigmoid-based cost with smooth transition"""
        if co2_level <= self.co2_target_ppm:
            return 0.0
        else:
            deviation = co2_level - self.co2_target_ppm
            # Sigmoid function that ramps up around target
            sigmoid = 1 / (1 + np.exp(-(deviation - 100) / 50))  # 100 ppm offset, 50 ppm scale
            return self.co2_weight * sigmoid * (deviation ** 1.5)
    
    def calculate_energy_cost(self, control_inputs: List[float], 
                            indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """
        Calculate total energy cost for all ventilation systems
        
        Args:
            control_inputs: Control inputs for each controllable ventilation system
            indoor_temp_c: Indoor temperature in Celsius
            outdoor_temp_c: Outdoor temperature in Celsius
            
        Returns:
            Total energy cost per second in dollars
        """
        total_cost_per_s = 0.0
        
        # Calculate cost for each controllable ventilation system
        for i, (ventilation_model, control_input) in enumerate(
            zip(self.room_dynamics.controllable_ventilations, control_inputs)
        ):
            # Get ventilation rate from control input
            ventilation_rate = ventilation_model.airflow_m3_per_hour(control_input)
            
            # Calculate energy cost for this ventilation system
            energy_cost_per_s = ventilation_model.energy_cost_per_s(
                ventilation_rate, indoor_temp_c, outdoor_temp_c
            )
            total_cost_per_s += energy_cost_per_s
        
        return total_cost_per_s
    
    def calculate_total_energy_cost(self, control_inputs: List[float], 
                                  indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """
        Calculate total energy cost including natural ventilation
        
        Args:
            control_inputs: Control inputs for each controllable ventilation system
            indoor_temp_c: Indoor temperature in Celsius
            outdoor_temp_c: Outdoor temperature in Celsius
            
        Returns:
            Total energy cost per second in dollars (including natural ventilation)
        """
        # Start with controllable ventilation costs
        total_cost_per_s = self.calculate_energy_cost(control_inputs, indoor_temp_c, outdoor_temp_c)
        
        # Add natural ventilation energy costs
        for natural_ventilation in self.room_dynamics.natural_ventilations:
            ventilation_rate = natural_ventilation.airflow_m3_per_hour()
            energy_cost_per_s = natural_ventilation.energy_cost_per_s(
                ventilation_rate, indoor_temp_c, outdoor_temp_c
            )
            total_cost_per_s += energy_cost_per_s
        
        return total_cost_per_s
    
    def cost_function(self, control_vector, current_co2_ppm, weather_conditions, indoor_temperature_c):
        """
        Calculate the total cost for a sequence of control inputs
        
        Args:
            control_vector: Flattened array of control inputs for all ventilation systems
            current_co2_ppm: Current indoor CO2 concentration
            weather_conditions: List of weather conditions over prediction horizon
            indoor_temperature_c: Indoor temperature
            
        Returns:
            Total cost (scalar)
        """
        # Reshape control vector into sequences for each ventilation system
        control_sequences = []
        for i in range(self.n_controllable):
            start_idx = i * self.n_steps
            end_idx = (i + 1) * self.n_steps
            control_sequences.append(control_vector[start_idx:end_idx].tolist())
        
        # Predict CO2 trajectory
        co2_trajectory = self.predict_co2_trajectory(
            current_co2_ppm, 
            control_sequences, 
            weather_conditions
        )
        
        total_cost = 0.0
        
        for i in range(self.n_steps):
            # CO2 cost
            co2_cost = self.co2_cost_function(co2_trajectory[i])
            
            # Energy cost for this time step
            control_inputs = [control_sequences[j][i] for j in range(self.n_controllable)]
            outdoor_temp = weather_conditions[i].outdoor_temperature
            energy_cost = self.calculate_energy_cost(
                control_inputs, indoor_temperature_c, outdoor_temp
            ) * self.step_size_seconds * self.energy_weight
            
            total_cost += co2_cost + energy_cost
        
        # Add control smoothness penalty
        if self.u_prev is not None:
            for i in range(self.n_controllable):
                control_change_penalty = self.control_smoothing_weight * (
                    control_sequences[i][0] - self.u_prev[i]
                ) ** 2
                total_cost += control_change_penalty
        
        # add a penalty for high ventilation rates to encourage smoothness
        # for i in range(self.n_controllable):
        #     control_change_penalty = self.control_smoothing_weight * (
        #         control_sequences[i][0]
        #     ) ** 2
        #     total_cost += control_change_penalty
    
        return total_cost
    
    def control(self, 
                current_co2_ppm: float,
                weather_conditions: List[WeatherConditions],
                indoor_temperature_c: float = 22.0) -> List[float]:
        """
        Compute optimal control inputs for all ventilation systems
        
        Args:
            current_co2_ppm: Current indoor CO2 concentration
            weather_conditions: Weather conditions over prediction horizon
            indoor_temperature_c: Indoor temperature
            
        Returns:
            List of optimal control inputs for each ventilation system
        """
        # Use Pontryagin's Maximum Principle for fastest convergence
        if self.optimization_method == "PMP":
            return self.pontryagin_optimization(current_co2_ppm, weather_conditions, indoor_temperature_c)
        
        # Ensure we have enough weather data
        if len(weather_conditions) < self.n_steps:
            last_weather = weather_conditions[-1] if weather_conditions else None
            while len(weather_conditions) < self.n_steps:
                if last_weather is not None:
                    weather_conditions.append(last_weather)
                else:
                    from src.models.weather import SolarIrradiation
                    default_solar = SolarIrradiation(0.0, 0.0, 0.0)
                    default_weather = WeatherConditions(default_solar, 0.0, 20.0, 15.0)
                    weather_conditions.append(default_weather)
        
        # Initial guess: keep rates steady
        if self.u_prev is not None:
            u0 = self.u_prev * self.n_steps
        else:
            u0 = np.zeros(self.n_controllable * self.n_steps)
        
        # Bounds: control inputs between 0 and maximum for each ventilation system
        bounds = []
        for i in range(self.n_controllable):
            max_rate = self.room_dynamics.controllable_ventilations[i].max_airflow_m3_per_hour
            bounds.extend([(0.0, max_rate) for _ in range(self.n_steps)])
        
        # Prepare optimization arguments
        args = (current_co2_ppm, weather_conditions, indoor_temperature_c)
        
        # Use analytical gradients if available and requested
        if self.use_analytical_gradients and self.optimization_method in ["SLSQP", "trust-constr"]:
            jac = lambda x: self.cost_function_gradient(x, *args)
        else:
            jac = None
        
        # Optimize with different methods
        if self.optimization_method == "SLSQP":
            result = optimize.minimize(
                self.cost_function,
                u0,
                args=args,
                method="SLSQP",
                jac=jac,
                bounds=bounds,
                options={'maxiter': self.max_iterations}
            )
        elif self.optimization_method == "trust-constr":
            result = optimize.minimize(
                self.cost_function,
                u0,
                args=args,
                method="trust-constr",
                jac=jac,
                bounds=bounds,
                options={'maxiter': self.max_iterations}
            )
        else:  # L-BFGS-B or other methods
            result = optimize.minimize(
                self.cost_function,
                u0,
                args=args,
                method=self.optimization_method,
                bounds=bounds,
                options={'maxiter': self.max_iterations}
            )
        
        if result.success:
            optimal_controls = []
            for i in range(self.n_controllable):
                start_idx = i * self.n_steps
                optimal_controls.append(result.x[start_idx])
        else:
            # Fallback: use the best solution found
            if hasattr(result, 'x') and result.x is not None:
                optimal_controls = []
                for i in range(self.n_controllable):
                    start_idx = i * self.n_steps
                    optimal_controls.append(result.x[start_idx])
            else:
                # Last resort: use moderate ventilation rates
                optimal_controls = [50.0] * self.n_controllable
        
        self.u_prev = optimal_controls
        return optimal_controls
    
    def get_control_info(self, 
                        current_co2_ppm: float,
                        weather_conditions: List[WeatherConditions],
                        indoor_temperature_c: float = 22.0) -> Tuple[List[float], List[float], float, Dict[str, Any]]:
        """
        Get detailed control information including predicted trajectory
        
        Args:
            current_co2_ppm: Current indoor CO2 concentration
            weather_conditions: Weather conditions over prediction horizon
            indoor_temperature_c: Indoor temperature
            
        Returns:
            Tuple of (optimal_controls, predicted_co2_trajectory, total_cost, additional_info)
        """
        # Ensure we have enough weather data
        if len(weather_conditions) < self.n_steps:
            last_weather = weather_conditions[-1] if weather_conditions else None
            while len(weather_conditions) < self.n_steps:
                if last_weather is not None:
                    weather_conditions.append(last_weather)
                else:
                    from src.models.weather import SolarIrradiation
                    default_solar = SolarIrradiation(0.0, 0.0, 0.0)
                    default_weather = WeatherConditions(default_solar, 0.0, 20.0, 15.0)
                    weather_conditions.append(default_weather)
        
        # Initial guess: moderate ventilation rates
        if self.u_prev is not None:
            u0 = self.u_prev * self.n_steps
        else:
            u0 = np.zeros(self.n_controllable * self.n_steps)
        
        # Bounds: control inputs between 0 and maximum for each ventilation system
        bounds = []
        for i in range(self.n_controllable):
            max_rate = self.room_dynamics.controllable_ventilations[i].max_airflow_m3_per_hour
            bounds.extend([(0.0, max_rate) for _ in range(self.n_steps)])
        
        # Optimize to get the full optimal sequence
        result = optimize.minimize(
            self.cost_function,
            u0,
            args=(current_co2_ppm, weather_conditions, indoor_temperature_c),
            method=self.optimization_method,
            bounds=bounds,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            optimal_sequence = result.x
            optimal_controls = []
            for i in range(self.n_controllable):
                start_idx = i * self.n_steps
                optimal_controls.append(optimal_sequence[start_idx])
            total_cost = result.fun
        else:
            # Fallback: use the best solution found
            if hasattr(result, 'x') and result.x is not None:
                optimal_sequence = result.x
                optimal_controls = []
                for i in range(self.n_controllable):
                    start_idx = i * self.n_steps
                    optimal_controls.append(optimal_sequence[start_idx])
                total_cost = result.fun
            else:
                # Last resort: use moderate ventilation rates
                optimal_controls = [50.0] * self.n_controllable
                optimal_sequence = np.ones(self.n_controllable * self.n_steps) * 50.0
                total_cost = self.cost_function(optimal_sequence, current_co2_ppm, weather_conditions, indoor_temperature_c)
        
        # Reshape optimal sequence for trajectory prediction
        control_sequences = []
        for i in range(self.n_controllable):
            start_idx = i * self.n_steps
            end_idx = (i + 1) * self.n_steps
            control_sequences.append(optimal_sequence[start_idx:end_idx].tolist())
        
        # Get predicted trajectory
        predicted_trajectory = self.predict_co2_trajectory(
            current_co2_ppm, 
            control_sequences, 
            weather_conditions[:self.n_steps]
        )
        
        # Calculate additional information
        additional_info = {
            'optimization_success': result.success,
            'optimization_message': result.message,
            'ventilation_rates_m3_per_hour': [],
            'energy_costs_per_s': []
        }
        
        # Calculate ventilation rates and energy costs for the first time step
        for i, (ventilation_model, control_input) in enumerate(
            zip(self.room_dynamics.controllable_ventilations, optimal_controls)
        ):
            ventilation_rate = ventilation_model.airflow_m3_per_hour(control_input)
            additional_info['ventilation_rates_m3_per_hour'].append(ventilation_rate)
            
            outdoor_temp = weather_conditions[0].outdoor_temperature
            energy_cost_per_s = ventilation_model.energy_cost_per_s(
                ventilation_rate, indoor_temperature_c, outdoor_temp
            )
            additional_info['energy_costs_per_s'].append(energy_cost_per_s)
        
        # Update previous control for next iteration
        self.u_prev = optimal_controls
        
        return optimal_controls, predicted_trajectory, total_cost, additional_info 

    def cost_function_gradient(self, control_vector, current_co2_ppm, weather_conditions, indoor_temperature_c):
        """
        Calculate analytical gradient of the cost function
        
        Args:
            control_vector: Flattened array of control inputs for all ventilation systems
            current_co2_ppm: Current indoor CO2 concentration
            weather_conditions: List of weather conditions over prediction horizon
            indoor_temperature_c: Indoor temperature
            
        Returns:
            Gradient of total cost with respect to control inputs
        """
        # Reshape control vector into sequences for each ventilation system
        control_sequences = []
        for i in range(self.n_controllable):
            start_idx = i * self.n_steps
            end_idx = (i + 1) * self.n_steps
            control_sequences.append(control_vector[start_idx:end_idx].tolist())
        
        # Predict CO2 trajectory
        co2_trajectory = self.predict_co2_trajectory(
            current_co2_ppm, 
            control_sequences, 
            weather_conditions
        )
        
        # Initialize gradient
        gradient = np.zeros_like(control_vector)
        
        # Calculate gradients for each time step
        for i in range(self.n_steps):
            # CO2 cost gradient
            co2_level = co2_trajectory[i]
            if self.co2_cost_type == "asymmetric_quadratic" and co2_level > self.co2_target_ppm:
                co2_gradient = 2 * self.co2_weight * (co2_level - self.co2_target_ppm)
            else:
                co2_gradient = 0.0
            
            # Energy cost gradient (simplified - assumes linear relationship)
            outdoor_temp = weather_conditions[i].outdoor_temperature
            temp_diff = indoor_temperature_c - outdoor_temp
            
            for j in range(self.n_controllable):
                idx = j * self.n_steps + i
                
                # Energy gradient (simplified)
                energy_gradient = self.energy_weight * self.step_size_seconds * 0.15 / 3600  # Simplified
                
                # Control smoothing gradient
                smoothing_gradient = 0.0
                if self.u_prev is not None and i == 0:
                    smoothing_gradient = 2 * self.control_smoothing_weight * (
                        control_sequences[j][0] - self.u_prev[j]
                    )
                
                gradient[idx] = co2_gradient + energy_gradient + smoothing_gradient
        
        return gradient
    
    def pontryagin_optimization(self, current_co2_ppm, weather_conditions, indoor_temperature_c):
        """
        Use Pontryagin's Maximum Principle for optimization
        This is much faster for this type of problem
        
        Args:
            current_co2_ppm: Current indoor CO2 concentration
            weather_conditions: List of weather conditions over prediction horizon
            indoor_temperature_c: Indoor temperature
            
        Returns:
            Optimal control inputs
        """
        # Simplified PMP approach - solve the optimality conditions directly
        # For linear-quadratic problems, this gives analytical solutions
        
        optimal_controls = []
        
        for i in range(self.n_controllable):
            # Simplified optimal control law based on current CO2 level
            if current_co2_ppm > self.co2_target_ppm:
                # High CO2 - use maximum ventilation
                optimal_control = self.room_dynamics.controllable_ventilations[i].max_airflow_m3_per_hour
            elif current_co2_ppm > self.co2_target_ppm - 50:
                # Near target - moderate ventilation
                optimal_control = self.room_dynamics.controllable_ventilations[i].max_airflow_m3_per_hour * 0.5
            else:
                # Low CO2 - minimal ventilation
                optimal_control = 0.0
            
            # Apply energy cost considerations
            outdoor_temp = weather_conditions[0].outdoor_temperature if weather_conditions else 20.0
            temp_diff = indoor_temperature_c - outdoor_temp
            
            # Reduce ventilation if it's very cold/hot outside (energy cost consideration)
            if abs(temp_diff) > 10:
                optimal_control *= 0.3
            
            optimal_controls.append(optimal_control)
        
        return optimal_controls 