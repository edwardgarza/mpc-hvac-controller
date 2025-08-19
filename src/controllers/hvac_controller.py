#!/usr/bin/env python3
"""
Integrated HVAC Controller combining heating/cooling and ventilation control
"""

import base64
import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import bool_
import scipy.optimize as optimize
from typing import List, Tuple, Optional, Dict, Any

from src.models.building import BuildingModel
from src.controllers.ventilation.models import RoomCO2Dynamics
from src.utils.timeseries import TimeSeries
from src.utils.calendar import Calendar, RelativeScheduleTimeSeries


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
                 step_size_hours: float = 0.25,
                 optimization_method: str = "L-BFGS-B",
                 max_iterations: int = 500, 
                 use_boolean_occupant_comfort: bool = True,
                 use_soft_boundary_condition: bool = True, 
                 smooth_controls: bool = False,
                 dynamically_lengthen_step_sizes: bool = False): 
        """
        Initialize integrated HVAC controller
        
        Args:
            room_dynamics: CO2 dynamics model
            building_model: Building thermal model (includes heating/cooling)
            horizon_hours: Prediction horizon in hours
            co2_weight: Weight for CO2 deviation penalty
            energy_weight: Weight for energy cost
            comfort_weight: Weight for temperature comfort
            step_size_hours: Time step size
            optimization_method: Optimization method
            max_iterations: Maximum optimization iterations
            use_boolean_occupant_comfort: if occupancy is 0 don't care about temperature or CO2 levels
            use_soft_boundary_condition: have a higher cost for the the last point's to emulate a boundary condition
            dynamically_lengthen_step_sizes: use larger step sizes further in the future
        """
        self.room_dynamics = room_dynamics
        self.building_model = building_model
        self.horizon_hours = horizon_hours
        self.co2_weight = co2_weight
        self.energy_weight = energy_weight
        self.comfort_weight = comfort_weight
        self.step_size_hours = step_size_hours
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.use_boolean_occupant_comfort = use_boolean_occupant_comfort
        self.use_soft_boundary_condition = use_soft_boundary_condition
        self.dynamically_lengthen_step_sizes = dynamically_lengthen_step_sizes
        self.smooth_controls = smooth_controls
        # Time discretization
        self.step_size_seconds = step_size_hours * 3600
        self.steps, self.cumulative_steps = self.generate_time_steps(self.step_size_hours, self.horizon_hours,  self.dynamically_lengthen_step_sizes)
        self.n_steps = int(horizon_hours / step_size_hours)
        
        # Control dimensions
        self.n_ventilation = len(room_dynamics.controllable_ventilations)
        self.n_hvac = 1  # Single heating/cooling control
        self.n_controls = self.n_ventilation + self.n_hvac
        
        # Previous control for smoothing
        self.u_prev = None
        self.next_prediction = None
        
        # Saved schedule from config
        self.saved_schedule = None
        self.set_points: RelativeScheduleTimeSeries = None 
        self._start_time: datetime = None 
    
    def set_saved_schedule(self, schedule):
        """Set the saved schedule from config"""
        self.saved_schedule = schedule
        self.calendar = Calendar(self.saved_schedule)
    
    
    def predict_trajectories(self, 
                           initial_co2_ppm: float,
                           initial_temp_c: float,
                           ventilation_control_sequences: List[List[float]],
                           hvac_control_sequences: List[List[float]],
                           weather_series_hours: TimeSeries,
                           start_time_hours_offset: float) -> Tuple[List[float], List[float]]:
        """
        Predict both CO2 and temperature trajectories
        
        Args:
            initial_co2_ppm: Starting CO2 concentration
            initial_temp_c: Starting temperature inside
            control_sequences: Control sequences [ventilation_controls, hvac_controls]
            weather_series_hours: TimeSeries of weather conditions
            start_time_hours_offset: Starting time for prediction (hours from weather series start)
            
        Returns:
            Tuple of (co2_trajectory, temp_trajectory), including the current time
        """

        co2_trajectory = [initial_co2_ppm]
        temp_trajectory = [initial_temp_c]
        
        current_co2 = initial_co2_ppm
        current_temp = initial_temp_c
        
        ventilation_controls = ventilation_control_sequences
        hvac_controls = hvac_control_sequences
        self.weather_series_hours = weather_series_hours
        for i in range(self.n_steps):
            # Extract control inputs for this time step
            ventilation_inputs = [ventilation_controls[j][i] for j in range(self.n_ventilation)]
            hvac_inputs = [hvac_controls[j][i] for j in range(self.n_hvac)]
            
            # Get weather at current prediction time
            # current_time_offset = i * self.step_size_hours + start_time_hours_offset
            current_time_offset = i * self.step_size_hours + 0
            weather = weather_series_hours.interpolate(current_time_offset)
            
            # use linear dynamics for better convergence
            # current_co2 = self.room_dynamics.co2_levels_in_t(current_co2, ventilation_inputs, self.step_size_seconds)
            current_co2 = self.room_dynamics.co2_change_per_s(current_co2, ventilation_inputs) * self.step_size_seconds + current_co2
            
            # Predict temperature change using building model
            # Calculate ventilation heat load (additional to building model)
            ventilation_heat_load_kw = 0.0
            for j, (vent_model, vent_input) in enumerate(
                zip(self.room_dynamics.controllable_ventilations, ventilation_inputs)
            ):
                ventilation_heat_load_kw += vent_model.energy_load_kw(
                    vent_input, current_temp, weather.outdoor_temperature
                )

            # Calculate natural ventilation heat load
            for natural_vent in self.room_dynamics.natural_ventilations:
                ventilation_heat_load_kw += natural_vent.energy_load_kw(
                    natural_vent.airflow_m3_per_hour(), current_temp, weather.outdoor_temperature
                )

            # Use building model for temperature change (includes HVAC and thermal transfer)
            temp_change_per_s = self.building_model.temperature_change_per_s(
                current_temp, weather, hvac_inputs[0], ventilation_heat_load_kw * 1000
            )

            temp_change = temp_change_per_s * self.step_size_seconds
            current_temp += temp_change

            # Store trajectories
            co2_trajectory.append(current_co2)
            temp_trajectory.append(current_temp)
        
        return co2_trajectory, temp_trajectory
    
    def _occupancy_comfort_cost_mult(self, time_hours_offset):
        return 1 if not self.use_boolean_occupant_comfort else self.set_points.interpolate_step_occupancy_count(time_hours_offset) > 0


    def comfort_cost(self, temperature_c: float, time_hours_offset: float = 0.0) -> Tuple[bool, float]:
        """
        Calculate comfort cost based on temperature deviation from target
        
        Args:
            temperature_c: Current temperature
            time_hours: Current time for dynamic setpoint lookup
            
        Returns:
            Tuple(above_temp, Comfort cost)
        """
        # Use dynamic setpoints if provided, otherwise use saved schedule
        target_temp = self.set_points.interpolate_step_temp(time_hours_offset)
        deviation = abs(temperature_c - target_temp)
        return (temperature_c - target_temp > 0, self.comfort_weight * (deviation ** 2) * self._occupancy_comfort_cost_mult(time_hours_offset))
    
    def co2_cost(self, co2_ppm: float, time_hours_offset: float = 0.0) -> float:
        """
        Calculate CO2 cost
        
        Args:
            co2_ppm: Current CO2 concentration
            time_hours: Current time for dynamic setpoint lookup
            
        Returns:
            CO2 cost
        """
        target_co2 = self.set_points.interpolate_step_co2(time_hours_offset)        
        deviation = max(0, co2_ppm - target_co2)
        return self.co2_weight * (deviation ** 2) * self._occupancy_comfort_cost_mult(time_hours_offset)
    
    def energy_cost(
            self, 
            ventilation_inputs: List[float], 
            hvac_input: float,
            time_hours: float = 0.0) -> float:
        """
        Calculate total energy cost for ventilation and HVAC. Note that this is not the same as the energy cost of the building model.

        Ventillation's only direct energy cost is the fan power, while hvac's costs is running the unit.
        
        Args:
            ventilation_inputs: Ventilation control inputs
            hvac_input: Heating/cooling control input (positive=heating, negative=cooling)
            time_hours: Current time for dynamic cost lookup
            
        Returns:
            Total energy cost per second
        """
        # Use dynamic costs if provided, otherwise use saved schedule
        electricity_cost = self.set_points.interpolate_step_energy_cost(time_hours)
        
        total_cost_per_s = 0.0
        
        # Ventilation energy cost
        for (vent_model, vent_input) in zip(self.room_dynamics.controllable_ventilations, ventilation_inputs):
            total_cost_per_s += vent_model.fan_power_w(vent_input) * electricity_cost / 3600 / 1000
                
        total_cost_per_s += abs(hvac_input) * electricity_cost / 3600 / 1000
        
        return total_cost_per_s
    
    def cost_function(
        self, 
        control_vector: np.ndarray,
        current_co2_ppm: float,
        current_temp_c: float,
        weather_series_hours: TimeSeries,
        start_time_hours_offset: float) -> float:
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
        ventilation_sequences, hvac_sequences = self.convert_flat_controls_to_ventillation_and_hvac(control_vector, 0)
        
        # Predict trajectories - note that co2 and temp trajectories include the starting step
        co2_trajectory, temp_trajectory = self.predict_trajectories(
                                            current_co2_ppm, 
                                            current_temp_c, 
                                            ventilation_sequences, 
                                            hvac_sequences, 
                                            weather_series_hours, 
                                            start_time_hours_offset
                                        )

        self.co2_trajectory = co2_trajectory
        self.temp_trajectory = temp_trajectory
        total_cost = 0.0
        accumulated_temp_error = 0
        accumulated_co2_error = 0
        last_step_above_temp = True
        for i in range(self.n_steps):
            # CO2 cost
            current_time_offset = i * self.step_size_hours + start_time_hours_offset
            co2_cost = self.co2_cost(co2_trajectory[i + 1], current_time_offset) * self.step_size_seconds
            if co2_cost > 0:
                accumulated_co2_error += co2_cost / 10
            else:
                accumulated_co2_error = 0
            # Comfort cost
            above_temp, comfort_cost_per_s = self.comfort_cost(temp_trajectory[i + 1], current_time_offset) 
            comfort_cost = comfort_cost_per_s * self.step_size_seconds
            if above_temp == last_step_above_temp:
                accumulated_temp_error += comfort_cost / 10
            else: 
                accumulated_temp_error = 0        
            last_step_above_temp = above_temp

            # Energy cost
            ventilation_inputs = [ventilation_sequences[j][i] for j in range(self.n_ventilation)]
            hvac_input = hvac_sequences[0][i]
            weather = weather_series_hours.interpolate(current_time_offset)
            outdoor_temp = weather.outdoor_temperature
            
            energy_cost = self.energy_cost(
                ventilation_inputs, 
                hvac_input,
                current_time_offset
            ) * self.step_size_seconds * self.energy_weight
            
            cost_of_step = co2_cost + comfort_cost + energy_cost + accumulated_temp_error + accumulated_co2_error
            total_cost += cost_of_step

        # double count last step's cost
        if self.use_soft_boundary_condition:
            total_cost += cost_of_step

        if self.smooth_controls:
            total_cost += self.cost_smoothed_controls(ventilation_sequences, hvac_sequences)

        return total_cost

    def cost_smoothed_controls(self, ventilation_sequences, hvac_sequences):
        '''
        Extra cost associated with changing control values between time steps - may make optimizations take longer to due 
        non-linearity.
        '''
        total_costs = 0
        for i in range(1, self.n_steps):
                for vent in range(self.n_ventilation):
                    control_change = ventilation_sequences[vent][i] - ventilation_sequences[vent][i - 1]
                    total_costs += control_change ** 2
                for h in range(self.n_hvac):
                    hvac_change = hvac_sequences[h][i] - hvac_sequences[h][i - 1]
                    total_costs += 0.1 * (hvac_change ** 2)

        # Control smoothing penalty from last executed value
        if self.u_prev is not None:
            for i in range(self.n_ventilation):
                control_change = ventilation_sequences[i][0] - self.u_prev[i]
                total_cost += control_change ** 2
            
            for h in range(self.n_hvac):
                hvac_change = hvac_sequences[h][0] - self.u_prev[self.n_ventilation + h]
                total_cost += 0.1 * (hvac_change ** 2)
        return total_costs / 100_000

    def optimize_controls(self,
                         current_co2_ppm: float,
                         current_temp_c: float,
                         weather_series_hours: TimeSeries,
                         start_time: datetime) -> Tuple[List[float], List[float], float]:
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
        max_time_needed = self.horizon_hours
        if max_time_needed > weather_series_hours.ticks[-1]:
            print(f"Warning: Weather forecast only goes to {weather_series_hours.ticks[-1]} hours, but need {max_time_needed} hours")
            # Use the last available weather for extrapolation
        self._start_time = start_time
        self.set_points = self.calendar.get_relative_schedule(start_time)

        # Initial guess
        if self.u_prev is not None:
            u0 = self.next_prediction
        else:
            u0 = np.zeros(self.n_controls * self.n_steps)
        
        # Bounds
        bounds = []
        # Ventilation bounds
        for i in range(self.n_ventilation):
            max_rate = self.room_dynamics.controllable_ventilations[i].max_airflow_m3_per_hour
            bounds.extend([(0.0, max_rate) for _ in range(self.n_steps)])
        # HVAC bounds (heating and cooling)
        bounds.extend([self.building_model.heating_model.output_range for _ in range(self.n_steps)])
        
        # Optimize
        result = optimize.minimize(
            self.cost_function,
            u0,
            args=(current_co2_ppm, current_temp_c, weather_series_hours, 0),
            method=self.optimization_method,
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'disp': True},
        )

        if not result.success:
            print(f"Optimization failed: {result.message}")
        ventilation_controls, hvac_controls = self.convert_flat_controls_to_ventillation_and_hvac(result.x)
        self.next_prediction = []
        for idx, val in enumerate(result.x[1:]):
            # peel off the first value in each control sequence to step forward in time
            if idx > 0 and idx % self.n_steps == 0:
                self.next_prediction.append(0)
            else:
                self.next_prediction.append(val)
        self.next_prediction.append(0)
        
        current_step_ventilation = [x[0] for x in ventilation_controls]
        current_step_hvac = [x[0] for x in hvac_controls]
        # Extract optimal controls or best guess if optimization fails
        total_cost = result.fun

        # Store for next iteration
        self.u_prev = current_step_ventilation + current_step_hvac
        self.optimized_ventilation_controls = ventilation_controls
        self.optimized_hvac_controls = hvac_controls
        # remove the current time step and append something random to the current optimized control for the next initial guess for the optimization
        return current_step_ventilation, current_step_hvac, total_cost
    
    def get_next_prediction(self) -> Optional[np.ndarray]:
        return self.next_prediction

    def get_optimized_hvac_controls(self) -> List[List[float]]:
        return self.optimized_hvac_controls

    def get_optimized_ventilation_controls(self) -> List[List[float]]:
        return self.optimized_ventilation_controls

    def get_structured_controls_next_step(self) -> Dict[str, Any]:
        ventilation_controls = self.get_optimized_ventilation_controls()        
        hvac_controls = self.get_optimized_hvac_controls()
        ventilation_dict = {}
        for i in range(self.n_ventilation):
            ventilation_dict[type(self.room_dynamics.controllable_ventilations[i]).__name__.replace("VentilationModel", "")] = ventilation_controls[i][0]
        hvac_dict = {}
        hvac_dict[type(self.building_model.heating_model).__name__.replace("ThermalDeviceModel", "")] = hvac_controls[0][0]

        return {   
            "co2_trajectory": self.get_co2_trajectory(),
            "temp_trajectory": self.get_temp_trajectory(),
            "hvac_dict": hvac_dict, 
            "ventilation_dict": ventilation_dict, 
            "estimated_cost": self.energy_costs_controls(ventilation_controls, hvac_controls),
            "pid_cost": self.energy_costs_hvac_pid(self.get_temp_trajectory()[0])}
    
    def get_temp_trajectory(self) -> List[float]:
        return self.temp_trajectory

    def get_co2_trajectory(self) -> List[float]:
        return self.co2_trajectory

    def get_start_time(self) -> datetime:
        return self._start_time         
    
    def generate_plot(self):
        x_axis = []
        set_point_temp = []
        set_point_co2 = []
        energy_cost = []
        outdoor_temp = []
        
        n_ventilation_vars = self.n_ventilation * self.n_steps
        print(f"[DEBUG] n_ventilation: {self.n_ventilation}, n_steps: {self.n_steps}, n_ventilation_vars: {n_ventilation_vars}")
        ventilation_sequences = self.get_optimized_ventilation_controls()
        hvac_sequence = self.get_optimized_hvac_controls()[0]
        print(f"[DEBUG] ventilation_vector len: {len(ventilation_sequences)}, hvac_vector len: {len(hvac_sequence)}")


        for step in range(self.n_steps + 1):
            relative_time_delta = self.step_size_hours * step
            time = self.get_start_time() + datetime.timedelta(seconds=self.step_size_seconds * step)
            x_axis.append(time)
            set_point_temp.append(self.set_points.interpolate_step_temp(relative_time_delta))
            set_point_co2.append(self.set_points.interpolate_step_co2(relative_time_delta))
            energy_cost.append(self.set_points.interpolate_step_energy_cost(relative_time_delta))
            outdoor_temp.append(self.weather_series_hours.interpolate(relative_time_delta).outdoor_temperature)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.set_title('Temperature and HVAC controls')
        ax1_control = ax1.twinx()
        ax1_control.plot(x_axis[1:], hvac_sequence, label="HVAC Controls", linestyle='--',)
        ax1_control.set_ylabel('Heating/Cooling Control Value (W)', color='red')
        ax1.plot(x_axis, set_point_temp, label="Set Point Temp", color='green')
        ax1.plot(x_axis, self.get_temp_trajectory(), label="Indoor Temp", color='orange')
        ax1.plot(x_axis, outdoor_temp, label="Outdoor Temp", color="blue")
        ax1.set_ylabel("Temperature")
        ax1.legend(loc='upper left')
        ax1_control.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2_vent = ax2.twinx()
        ax2.set_ylabel("CO2 PPM")
        ax2.set_title('CO₂ Levels and Ventilation Controls')
        ax2_vent.set_ylabel('Ventilation Control Value (m^3/hr)', color='purple')
        ax2.plot(x_axis, self.get_co2_trajectory(), label="CO2 Levels", color='orange')
        ax2.plot(x_axis, set_point_co2, label="Set Point CO2", color='green')

        for i in range(len(ventilation_sequences)):
            ax2_vent.plot(x_axis[1:], ventilation_sequences[i], linestyle='--',
            label="Ventillation " + type(self.room_dynamics.controllable_ventilations[i]).__name__)
        ax2_vent.legend(loc='upper left')
        
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # plt.plot(x_axis, energy_cost, label="energy")
        plt.tight_layout()
        plt.show()
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"plot_data": plot_data}

    def energy_costs_hvac_pid(self, start_temp):
        '''
        Calculate what the energy costs would've been if only hvac  and natural ventilation is considered and a
         normal control scheme is used that is unaware of time of use pricing or home/away scheduling.
        '''
        cost = 0
        total_energy = 0
        set_point_temp = self.set_points.interpolate_step_temp(0)
        outdoor_weather = self.weather_series_hours.interpolate(0)
        initial_j = self.building_model.heat_capacity * (start_temp - set_point_temp)
        additional_energy_used_j = self.building_model.heating_model.power_consumed(initial_j, set_point_temp, outdoor_weather.outdoor_temperature)

        for step in range(self.n_steps + 1):
            relative_time_delta = self.step_size_hours * step
            set_point_temp = self.set_points.interpolate_step_temp(relative_time_delta)
            energy_cost = self.set_points.interpolate_step_energy_cost(relative_time_delta)
            outdoor_weather = self.weather_series_hours.interpolate(relative_time_delta)
            ventilation_load_kw = sum([x.energy_load_kw(None, set_point_temp, outdoor_weather.outdoor_temperature) for x in self.room_dynamics.natural_ventilations])
            heat_change = self.building_model.powerflow(set_point_temp, outdoor_weather) + ventilation_load_kw * 1000
            power_input = self.building_model.heating_model.power_consumed(-heat_change, set_point_temp, outdoor_weather.outdoor_temperature)
            energy =  power_input / 1000 * self.step_size_hours  + additional_energy_used_j / 3600 / 1000 # kwh + j
            additional_energy_used_j = 0
            step_cost = energy * energy_cost 
            cost += step_cost
            total_energy += energy
        return cost

    def energy_costs_controls(self, ventilation_controls: List[List[float]], hvac_controls: List[List[float]]) -> float:

        total_energy_cost = 0
        for i in range(self.n_steps):
            # CO2 cost
            current_time_offset = i * self.step_size_hours
            
            # Energy cost
            ventilation_inputs = [ventilation_controls[j][i] for j in range(self.n_ventilation)]

            # TODO: support multiple hvac units
            hvac_inputs = [hvac_controls[j][i] for j in range(self.n_hvac)]

            energy_cost = self.energy_cost(
                ventilation_inputs, 
                hvac_inputs[0],
                current_time_offset
            ) * self.step_size_seconds
            total_energy_cost += energy_cost
        return total_energy_cost

    def get_control_info(self,
                        current_co2_ppm: float,
                        current_temp_c: float,
                        weather_series_hours: TimeSeries,
                        start_time: datetime = 0.0) -> Tuple[List[float], List[float], List[float], List[float], float, Dict[str, Any]]:
        """
        Get detailed control information including predicted trajectories
        
        Args:
            current_co2_ppm: Current CO2 concentration
            current_temp_c: Current temperature
            weather_conditions: Weather conditions over horizon
            
        Returns:
            Tuple of (ventilation_controls, hvac_controls, co2_trajectory, temp_trajectory, total_cost, additional_info)
        """
        _ = self.optimize_controls(
            current_co2_ppm, current_temp_c, weather_series_hours, start_time
        )
        
        # Predict full trajectories
        ventilation_controls = self.get_optimized_ventilation_controls()        
        hvac_controls = self.get_optimized_hvac_controls()
        
        co2_trajectory, temp_trajectory = self.predict_trajectories(
                                            current_co2_ppm, 
                                            current_temp_c,
                                            ventilation_controls, 
                                            hvac_controls,
                                            weather_series_hours, 
                                            0
                                        )   
        return {
            "ventillation_controls": ventilation_controls,
            "hvac_controls": hvac_controls,
            "co2_trajectory": co2_trajectory,
            "temp_trajectory": temp_trajectory,
            "total_energy_cost_dollars": self.energy_costs_controls(ventilation_controls, hvac_controls),
            "energy_cost_dollars_pid": self.energy_costs_hvac_pid(current_temp_c)
        }

    def generate_time_steps(self, min_step_size_hours: float, horizon_hours: float, dynamically_lengthen_step_sizes: bool, double_size_every_x_points: int = 4):
        total_times = []
        steps = []
        mult = 1
        next_value = min_step_size_hours
        while next_value < horizon_hours:
            if not steps:
                total_times.append(min_step_size_hours)
                steps.append(min_step_size_hours)
            else:            
                steps.append(min_step_size_hours * mult)
                total_times.append(next_value)
            if dynamically_lengthen_step_sizes and len(steps) % double_size_every_x_points == 0:
                mult *= 2
            next_value = total_times[-1] + min_step_size_hours * mult

        return steps, total_times

    def convert_flat_controls_to_ventillation_and_hvac(self, control_inputs: List[float], start_offset: int = 0) -> Tuple[List[List[float]], List[List[float]]]:
        '''
        Convert flattened list of control inputs into a tuple of a list of conrol inputs.

        the input looks like [hvac0_t0, hvac1_t0..., hvacn_tend, ventilation0_t1, ventilation1_t0, ... ventilationn_tend]

        and the output is 

        Tuple([[hvac0_t0, hvac0_t1, ...hvac0_tend], ...[hvacn_t0..., hvacn_tend]], 
        [[ventilation0_t0, ventilation0_t1, ...ventilation0_tend], ...[ventilationn_t0..., ventilationn_tend]])
        '''
        ventilation_vector = control_inputs[:self.n_ventilation * self.n_steps]
        hvac_vector = control_inputs[self.n_ventilation * self.n_steps:]
        ventilation_controls = []
        for i in range(self.n_ventilation):
            start_idx = i * self.n_steps + start_offset
            ventilation_controls.append(ventilation_vector[start_idx: (i + 1) * self.n_steps])
        
        hvac_controls = []
        for i in range(self.n_hvac):
            start_idx = i * self.n_steps + start_offset
            hvac_controls.append(hvac_vector[start_idx: (i + 1) * self.n_steps])

        return ventilation_controls, hvac_controls
