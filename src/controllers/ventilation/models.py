#!/usr/bin/env python3
"""
Ventilation Models for different types of ventilation systems
"""

from abc import ABC, abstractmethod, abstractproperty
import math
from typing import List
from src.models.weather import WeatherConditions


class BaseVentilationModel(ABC):
    """
    Abstract base class for controllable ventilation models
    """
    
    def __init__(self):
        """
        Initialize base ventilation model
        
        Args:
        """

    @abstractmethod
    def energy_load_kw(self, ventilation_rate_m3_per_hour: float, 
                      indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """
        Calculate heating/cooling load on the building for this ventilation type. Negative is cooling, positive is heating.
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            indoor_temp_c: Indoor temperature in Celsius
            outdoor_temp_c: Outdoor temperature in Celsius
            
        Returns:
            Heating/cooling load in kW (positive = heating needed, negative = cooling needed)
            Note: Fan power is included in the load since fans are inside the building
        """
        pass
    
    @abstractmethod
    def fan_power_w(self, ventilation_rate_m3_per_hour: float) -> float:
        """
        Calculate fan power for this ventilation type
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            
        Returns:
            Fan power in W
        """
        pass
    
    @property
    @abstractmethod
    def max_airflow_m3_per_hour(self) -> float:
        """
        Maximum airflow rate for this ventilation system
        
        Returns:
            Maximum airflow in m³/hour
        """
        pass
    
    def energy_cost_per_s(self, ventilation_rate_m3_per_hour: float,
                         indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """
        Calculate energy cost for this ventilation type. This is only useful for the ventillation only controller.
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            indoor_temp_c: Indoor temperature in Celsius
            outdoor_temp_c: Outdoor temperature in Celsius
            
        Returns:
            Energy cost per second in dollars
        """
        # Calculate total energy load (including fan power)
        total_energy_load_kw = self.energy_load_kw(ventilation_rate_m3_per_hour, indoor_temp_c, outdoor_temp_c) 
        
        # Assume $0.15/kWh for electricity
        electricity_cost_per_kwh = 0.15
        
        # Convert to cost per second
        cost_per_s = abs(total_energy_load_kw) * electricity_cost_per_kwh / 3600
        
        # for now this is a little strange - we have the energy added to the building with an additional fan cost.
        cost_per_s += self.fan_power_w(ventilation_rate_m3_per_hour) / 1000 * electricity_cost_per_kwh / 3600
        return cost_per_s
    

    def airflow_m3_per_hour(self, control_input):
        pass

    def _base_energy_load_kw(self, ventilation_rate_m3_per_hour: float, 
                           indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """Calculate base energy load (same as window ventilation)"""
        # Air properties
        air_density_kg_per_m3 = 1.225
        specific_heat_j_per_kg_k = 1005
        
        ventilation_rate_m3_per_s = ventilation_rate_m3_per_hour / 3600
        
        # Temperature difference
        temp_diff_k = outdoor_temp_c - indoor_temp_c
        
        # Energy load in watts
        energy_load_w = ventilation_rate_m3_per_s * air_density_kg_per_m3 * specific_heat_j_per_kg_k * temp_diff_k
        
        # Convert to kW
        energy_load_kw = energy_load_w / 1000
        
        return energy_load_kw

class NaturalVentilationModel(BaseVentilationModel):
    """
    Model for uncontrollable natural ventilation (infiltration through building leaks)
    """
    
    def __init__(self, 
                 indoor_volume_m3: float,
                 infiltration_rate_ach: float = 0.1, 
                 effective_leak_area_m3: float = 0.0):
        """
        Initialize natural ventilation model
        
        Args:
            indoor_volume_m3: Indoor volume in cubic meters
            infiltration_rate_ach: Natural infiltration rate in air changes per hour
        """
        self.indoor_volume_m3 = indoor_volume_m3
        self.infiltration_rate_ach = infiltration_rate_ach
        self.efla = effective_leak_area_m3
        self.stack_coeficient_per_story = 0.000145 # default value
        self._max_airflow_m3_per_hour = infiltration_rate_ach * indoor_volume_m3

    @property
    def infiltration_flow_rate_m3_per_hour(self):
        """Infiltration flow rate in m³/hour"""
        return self.infiltration_rate_ach * self.indoor_volume_m3
    
    @property
    def max_airflow_m3_per_hour(self) -> float:
        """Maximum airflow rate for natural ventilation"""
        return self._max_airflow_m3_per_hour

    def airflow_m3_per_hour(self, _ : float = 0.0):
        return self.infiltration_rate_ach * self.indoor_volume_m3

    # def airflow_m3_per_hour(self, indoor_temp : float, outdoor_weather: WeatherConditions):
    #     return self.efla / 10 ** 6 * 3.6 * math.sqrt(abs(indoor_temp - outdoor_weather.outdoor_temperature)) * self.stack_coeficient_per_story

    def energy_load_kw(self, _: float, 
                      indoor_temp_c: float, outdoor_temp_c: float) -> float:
        # Natural ventilation has energy cost due to heating/cooling infiltrated air
        # (no fan power, but temperature difference creates heating/cooling load)
        return self._base_energy_load_kw(self.airflow_m3_per_hour(0), indoor_temp_c, outdoor_temp_c)
    
    def fan_power_w(self, ventilation_rate_m3_per_hour: float) -> float:
        # Natural ventilation has no fan power
        return 0.0



class WindowVentilationModel(BaseVentilationModel):
    """
    Model for window-based ventilation (controllable natural ventilation)
    
    This represents opening windows or doors for natural ventilation.
    - No energy recovery
    - No fan power (natural convection)
    - Full temperature difference affects heating/cooling load
    - Simple CO2 dynamics
    """
    
    def __init__(self, max_airflow_m3_per_hour: float = 100):
        """
        Initialize window ventilation model
        
        Args:
            max_airflow_m3_per_hour: Maximum airflow rate in m³/hour
        """
        self._max_airflow_m3_per_hour = max_airflow_m3_per_hour
    
    def energy_load_kw(self, ventilation_rate_m3_per_hour: float, 
                      indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """
        Calculate heating/cooling load for window ventilation
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            indoor_temp_c: Indoor temperature in Celsius
            outdoor_temp_c: Outdoor temperature in Celsius
            
        Returns:
            Heating/cooling load in kW (positive = heating added, negative = heat removed)
            Note: Fan power is included in the load since fans are inside the building
        """
        # Air properties
        return self._base_energy_load_kw(ventilation_rate_m3_per_hour, indoor_temp_c, outdoor_temp_c)
    
    def fan_power_w(self, ventilation_rate_m3_per_hour: float) -> float:
        """
        Calculate fan power for window ventilation
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            
        Returns:
            Fan power in kW (zero for natural ventilation)
        """
        # Window ventilation uses natural convection, no fan power required
        return 0.0

    def airflow_m3_per_hour(self, window_opening_m3_per_hour):
        return window_opening_m3_per_hour
    
    @property
    def max_airflow_m3_per_hour(self) -> float:
        """Maximum airflow rate for window ventilation"""
        return self._max_airflow_m3_per_hour


class ERVVentilationModel(BaseVentilationModel):
    """
    Model for Energy Recovery Ventilator (ERV)
    
    ERV systems recover both heat and moisture from exhaust air.
    - Heat and moisture recovery efficiency reduces heating/cooling load
    - Fan power required for operation

    Moisture recovery is not implemented.
    """
    
    def __init__(self, 
        heat_recovery_efficiency: float = 0.8, 
        moisture_recovery_efficiency: float = 0.5, 
        fan_power_w_m3_per_hour: float = 0.5, 
        max_airflow_m3_per_hour: float = 100):
        """
        Initialize ERV model
        
        Args:
            heat_recovery_efficiency: Heat recovery efficiency (0.0 to 1.0)
            moisture_recovery_efficiency: Moisture recovery efficiency (0.0 to 1.0)
            fan_power_w_m3_per_hour: Fan power in watts per m³/hour of airflow
            max_airflow_m3_per_hour: Maximum airflow rate in m³/hour
        """
        self.heat_recovery_efficiency = heat_recovery_efficiency
        self.moisture_recovery_efficiency = moisture_recovery_efficiency
        self.fan_power_w_m3_per_hour = fan_power_w_m3_per_hour
        self._max_airflow_m3_per_hour = max_airflow_m3_per_hour
    
    def energy_load_kw(self, ventilation_rate_m3_per_hour: float, 
                      indoor_temp_c: float, outdoor_temp_c: float) -> float:
        """
        Calculate heating/cooling load for ERV with heat and moisture recovery
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            indoor_temp_c: Indoor temperature in Celsius
            outdoor_temp_c: Outdoor temperature in Celsius
            
        Returns:
            Heating/cooling load in kW (positive = heating needed, negative = cooling needed)
            Note: Fan power is included in the load since fans are inside the building
        """
        # Calculate base energy load (like window ventilation)
        base_energy_load_kw = self._base_energy_load_kw(ventilation_rate_m3_per_hour, indoor_temp_c, outdoor_temp_c)
        
        # Apply heat recovery efficiency
        # If heat recovery efficiency is 80%, we only need to provide 20% of the base load
        effective_energy_load_kw = base_energy_load_kw * (1 - self.heat_recovery_efficiency)
        
        # Add fan power to the load (fans are inside, so their heat contributes to the load)
        fan_power_w = self.fan_power_w(ventilation_rate_m3_per_hour)
        total_energy_load_kw = effective_energy_load_kw + fan_power_w / 1000
        
        return total_energy_load_kw
    
    def fan_power_w(self, ventilation_rate_m3_per_hour: float) -> float:
        """
        Calculate fan power for ERV operation
        
        Args:
            ventilation_rate_m3_per_hour: Ventilation rate in m³/hour
            
        Returns:
            Fan power in kW
        """
        # Convert to m³/s
        return self.fan_power_w_m3_per_hour * ventilation_rate_m3_per_hour
        
    def airflow_m3_per_hour(self, fan_setting_m3_per_hour):
        return fan_setting_m3_per_hour
    
    @property
    def max_airflow_m3_per_hour(self) -> float:
        """Maximum airflow rate for ERV"""
        return self._max_airflow_m3_per_hour


class HRVVentilationModel(ERVVentilationModel):
    """
    Model for Heat Recovery Ventilator (HRV)
    
    HRV systems recover heat from exhaust air to preheat incoming fresh air.
    - Heat recovery efficiency reduces heating/cooling load
    - No moisture recovery (unlike ERV)
    - Fan power required for operation
    """
    
    def __init__(self, 
        heat_recovery_efficiency: float = 0.7, 
        fan_power_w_m3_per_hour: float = 0.5, 
        max_airflow_m3_per_hour: float = 100):
        """
        Initialize HRV model
        
        Args:
            heat_recovery_efficiency: Heat recovery efficiency (0.0 to 1.0)
            fan_power_w_m3_per_hour: Fan power in watts per m³/hour of airflow
        """
        # HRV has no moisture recovery, so set moisture_recovery_efficiency to 0.0
        super().__init__(heat_recovery_efficiency=heat_recovery_efficiency, 
                        moisture_recovery_efficiency=0.0, 
                        fan_power_w_m3_per_hour=fan_power_w_m3_per_hour,
                        max_airflow_m3_per_hour=max_airflow_m3_per_hour)


class RoomCO2Dynamics:
    """
    Combines sources and ventilation models to compute net CO2 change in the room.
    """
    def __init__(
        self, 
        volume_m3: float, 
        controllable_ventilations :List[BaseVentilationModel], 
        natural_ventilations: List[NaturalVentilationModel], 
        outdoor_co2_ppm: float = 400):
        self.volume_m3 = volume_m3
        self.controllable_ventilations = controllable_ventilations  # list of ventilation models
        self.natural_ventilations = natural_ventilations  # list of ventilation models
        self.outdoor_co2_ppm = outdoor_co2_ppm
    
    def co2_change_per_s(self, co2_ppm, control_inputs, co2_production_m_3_hr):
        # control_inputs: list of control inputs for each controllable ventilation model
        total_airflow = 0.0
        
        # Add controllable ventilation
        for v, u in zip(self.controllable_ventilations, control_inputs):
            total_airflow += v.airflow_m3_per_hour(u)
        
        # Add natural ventilation (no control input needed)
        for v in self.natural_ventilations:
            total_airflow += v.airflow_m3_per_hour()
        
        # Calculate air exchange rate (fraction of room air replaced per second)
        air_exchange_per_s = total_airflow / 3600 / self.volume_m3
        
        # CO2 production rate (ppm/s)
        production_ppm_per_s = (co2_production_m_3_hr / self.volume_m3) * 1e6 / 3600
        
        # CO2 removal rate (ppm/s) - exponential decay
        # The rate of CO2 removal is proportional to the current concentration difference
        removal_ppm_per_s = air_exchange_per_s * (co2_ppm - self.outdoor_co2_ppm)
        
        # Net CO2 change rate
        net_co2_change_ppm_per_s = production_ppm_per_s - removal_ppm_per_s
        
        return net_co2_change_ppm_per_s 
