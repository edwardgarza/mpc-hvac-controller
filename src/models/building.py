"""
Building thermal models for HVAC control simulation.

This module provides classes for modeling the thermal behavior of building components
including walls, windows, roofs, floors, and complete building assemblies.

All units are in SI (metric):
- Temperatures: Celsius
- Distances: meters
- Areas: square meters
- Power: watts
- Energy: joules
- Thermal resistance: m²·K/W
- Heat capacity: J/K
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np

from src.utils.orientation import Orientation
from src.models.weather import WeatherConditions
from src.models.thermal_transfer import ThermalTransfer
from src.models.thermal_device import ThermalDeviceModel, ElectricResistanceThermalDeviceModel


class Studs:
    """
    Model for wooden studs in wall construction.
    
    Attributes:
        width: Width of the stud in meters (typically 0.038 m for 2x4)
        depth: Depth of the stud in meters (typically 0.089 m for 2x4)
        spacing: Center-to-center spacing between studs in meters (typically 0.406 m for 16 inch spacing)
    """
    
    def __init__(self, width: float, depth: float, spacing: float) -> None:
        """
        Initialize stud configuration.
        
        Args:
            width: Width of the stud in meters
            depth: Depth of the stud in meters
            spacing: Center-to-center spacing between studs in meters
        """
        self.width = width
        self.depth = depth
        self.spacing = spacing


class WallModel(ThermalTransfer):
    """
    Thermal model for a wall assembly including studs and insulation.
    
    Models heat transfer through walls including conduction through studs,
    insulation, and solar absorption effects.
    """
    
    def __init__(self, studs: Studs, insulation_r: float, 
                 area_sq_m: float, orientation: Orientation, exterior_insulation_r: float = 0.0) -> None:
        """
        Initialize wall model.
        
        Args:
            studs: Stud configuration for the wall
            insulation_r: R-value of insulation in m²·K/W
            area_sq_m: Wall area in square meters
            orientation: Wall orientation (North, South, East, West)
            exterior_insulation_r: R-value of exterior insulation in m²·K/W (purely additive)
        """
        self.studs = studs
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m
        self.orientation = orientation
        self.exterior_insulation_r = exterior_insulation_r

    def _rvalue(self) -> float:
        """
        Calculate effective R-value of the wall assembly.
        
        Uses weighted average of stud and insulation R-values based on
        the proportion of each in the wall, plus any exterior insulation.
        
        Returns:
            Effective R-value of the wall assembly in m²·K/W
        """
        # Wood thermal conductivity is approximately 0.12 W/m·K
        # R-value = thickness / thermal_conductivity
        r_value_studs = self.studs.depth / 0.12  # m²·K/W of wood studs
        # Weighted sum of the R values based on stud spacing
        base_r = (self.studs.width * r_value_studs + 
                  (self.studs.spacing - self.studs.width) * self.insulation_r) / self.studs.spacing
        return base_r + self.exterior_insulation_r

    def powerflow(self, inside_temperature: float, 
                  weather_conditions: WeatherConditions) -> float:
        """
        Calculate heat flow through the wall.
        
        Includes both conductive heat transfer and solar absorption effects.
        
        Args:
            inside_temperature: Indoor temperature in Celsius
            weather_conditions: Current weather conditions including solar data
            
        Returns:
            Heat flow in watts (positive = heat flowing into building)
        """
        # Calculate solar temperature with 0.5 absorptivity (typical for walls)
        solar_temp = weather_conditions.sol_temp(0.5, self.orientation)
        return (solar_temp - inside_temperature) * self.area_sq_m / self._rvalue()


class RoofModel(ThermalTransfer):
    """
    Thermal model for a roof assembly.
    
    Models heat transfer through the roof including insulation and solar absorption.
    """
    
    def __init__(self, insulation_r: float, area_sq_m: float, 
                 orientation: Orientation, absorptivity: float = 0.85) -> None:
        """
        Initialize roof model.
        
        Args:
            insulation_r: R-value of roof insulation in m²·K/W
            area_sq_m: Roof area in square meters
            orientation: Roof orientation
            absorptivity: Solar absorptivity of roof surface (0-1)
        """
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m
        self.absorptivity = absorptivity
        self.orientation = orientation

    def powerflow(self, inside_temperature: float, 
                  weather_conditions: WeatherConditions) -> float:
        """
        Calculate heat flow through the roof.
        
        Args:
            inside_temperature: Indoor temperature in Celsius
            weather_conditions: Current weather conditions including solar data
            
        Returns:
            Heat flow in watts (positive = heat flowing into building)
        """
        solar_temp = weather_conditions.sol_temp(self.absorptivity, self.orientation)
        return (solar_temp - inside_temperature) * self.area_sq_m / self.insulation_r


class FloorModel(ThermalTransfer, ABC):
    """Abstract base class for floor thermal models."""
    pass


class SlabModel(FloorModel):
    """
    Thermal model for a concrete slab floor.
    
    Models heat transfer through a concrete slab to the ground.
    """
    
    def __init__(self, insulation_r: float, area_sq_m: float) -> None:
        """
        Initialize slab floor model.
        
        Args:
            insulation_r: R-value of floor insulation in m²·K/W
            area_sq_m: Floor area in square meters
        """
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m

    def powerflow(self, inside_temperature: float, 
                  weather_conditions: WeatherConditions) -> float:
        """
        Calculate heat flow through the floor slab.
        
        Args:
            inside_temperature: Indoor temperature in Celsius
            weather_conditions: Current weather conditions (uses ground temperature)
            
        Returns:
            Heat flow in watts (positive = heat flowing into building)
        """
        return (weather_conditions.ground_temperature - inside_temperature) * self.area_sq_m / self.insulation_r


class PierAndBeam(WallModel, FloorModel):
    """
    Thermal model for a pier and beam floor system.
    
    This is essentially a wall model but with no solar irradiation effects.
    Models heat transfer through the floor to the outdoor air below.
    """
    def __init__(self, studs: Studs, insulation_r: float, area_sq_m: float, orientation: Orientation, exterior_insulation_r: float = 0.0) -> None:
        super().__init__(studs, insulation_r, area_sq_m, orientation, exterior_insulation_r)

    def powerflow(self, inside_temperature: float, 
                  weather_conditions: WeatherConditions) -> float:
        """
        Calculate heat flow through pier and beam floor.
        
        Args:
            inside_temperature: Indoor temperature in Celsius
            weather_conditions: Current weather conditions (uses outdoor temperature)
            
        Returns:
            Heat flow in watts (positive = heat flowing into building)
        """
        return (inside_temperature - weather_conditions.outdoor_temperature) * self.area_sq_m / self._rvalue()


class WindowModel(ThermalTransfer):
    """
    Thermal model for windows.
    
    Models both conductive heat transfer and solar heat gain through windows.
    """
    
    def __init__(self, insulation_r: float, area_sq_m: float, 
                 solar_heat_gain_coefficient: float, orientation: Optional[Orientation] = None) -> None:
        """
        Initialize window model.
        
        Args:
            insulation_r: R-value of window assembly in m²·K/W
            area_sq_m: Window area in square meters
            solar_heat_gain_coefficient: SHGC of window (0-1)
            orientation: Window orientation (optional)
        """
        self.insulation_r = insulation_r
        self.area_sq_m = area_sq_m
        self.shgc = solar_heat_gain_coefficient
        self.orientation = orientation

    def powerflow(self, inside_temperature: float, 
                  weather_conditions: WeatherConditions) -> float:
        """
        Calculate heat flow through windows.
        
        Includes both conductive heat transfer and solar heat gain.
        
        Args:
            inside_temperature: Indoor temperature in Celsius
            weather_conditions: Current weather conditions including solar data
            
        Returns:
            Heat flow in watts (positive = heat flowing into building)
        """
        # Thermal conduction and solar irradiance pass through
        conductive_heat = (weather_conditions.outdoor_temperature - inside_temperature) / self.insulation_r
        
        # Calculate solar heat gain if orientation is available
        if self.orientation is not None:
            solar_heat = self.shgc * weather_conditions.projected_intensity(self.orientation)
        else:
            solar_heat = 0.0  # No solar gain if no orientation specified
            
        return self.area_sq_m * (conductive_heat + solar_heat)


class BuildingModel(ThermalTransfer):
    """
    Complete building thermal model.
    
    Combines all building components (walls, windows, roof, floor) and
    the heating/cooling system to model the complete thermal behavior
    of a building.
    """
    
    """
    Initialize building model.
    
    Args:
        thermal_models: List of thermal models for walls, windows, roof, floor. Can be multiple of each.
        heating_model: Heating/cooling system model
        heat_capacity: Building heat capacity in J/K
    """
    def __init__(self, thermal_models: List[ThermalTransfer], heating_model: ThermalDeviceModel, heat_capacity: float, baseload_interior_heating: float = 0.0) -> None:
        self.thermal_models = thermal_models
        self.heating_model = heating_model
        self.heat_capacity = heat_capacity
        self.baseload_interior_heating = baseload_interior_heating

    def powerflow(self, *args: Any) -> float:
        """
        Calculate total heat flow into the building.
        
        Sums heat flow from all building components.
        
        Args:
            *args: Arguments passed to individual thermal models
                  (typically inside_temperature and weather_conditions)
            
        Returns:
            Total heat flow in watts (negative = heat flowing out of building)
        """
        total_power = self.baseload_interior_heating
        for model in self.thermal_models:
            power_change = model.powerflow(*args)
            if power_change is not None:
                total_power += power_change
        return total_power

    def temperature_change_per_s(self,
                                inside_temperature: float,
                                weather_conditions: WeatherConditions,
                                hvac_power_input: float, 
                                additional_power_input: float = 0.0) -> float:
        """
        Calculate instantaneous temperature change rate.
        
        This is the linear approximation method. For more accurate results,
        use integrate_temperature_change() instead.
        
        Args:
            inside_temperature: Current indoor temperature in Celsius
            weather_conditions: Current weather conditions
            hvac_power_input: HVAC power input in watts
            additional_power_input: Additional heat sources in watts
            
        Returns:
            Temperature change rate in °C/second
        """
        hvac_power_produced = self.heating_model.power_produced(
            hvac_power_input,
            inside_temperature,
            weather_conditions.outdoor_temperature
        )
        
        total_heat_flow = self.powerflow(inside_temperature, weather_conditions)
        return (total_heat_flow + hvac_power_produced + additional_power_input) / self.heat_capacity

    def integrate_temperature_change(self,
                                   initial_temperature: float,
                                   weather_conditions: WeatherConditions,
                                   hvac_power_input: float,
                                   time_seconds: float,
                                   additional_power_input: float = 0.0) -> float:
        """
        Properly integrate temperature change over a time step.
        
        Uses the exact solution of the temperature differential equation:
        dT/dt = (heat_flow + hvac_power) / heat_capacity
        
        This method is more accurate than the linear approximation for
        longer time steps or large temperature differences.
        
        Args:
            initial_temperature: Starting temperature in Celsius
            weather_conditions: Weather conditions during the time step
            hvac_power_input: HVAC power input in watts
            time_seconds: Time step duration in seconds
            additional_power_input: Additional heat sources in watts
            
        Returns:
            Final temperature in Celsius after the time step
        """
        # Get HVAC power output
        hvac_power = self.heating_model.power_produced(
            hvac_power_input,
            initial_temperature,
            weather_conditions.outdoor_temperature
        )
        
        # Calculate total heat flow at initial temperature
        total_heat_flow = self.powerflow(initial_temperature, weather_conditions)
        
        # Calculate heat transfer coefficient U
        temp_diff = weather_conditions.outdoor_temperature - initial_temperature
        if abs(temp_diff) > 0.1:
            U = total_heat_flow / temp_diff
        else:
            U = 0.0
            
        # Solve the differential equation
        if abs(U) < 1e-6:
            # No heat transfer - simple linear integration
            total_power = hvac_power + additional_power_input
            temp_change = total_power * time_seconds / self.heat_capacity
            return initial_temperature + temp_change
        else:
            # Heat transfer present - use exponential solution
            # T(t) = T_steady + (T₀ - T_steady) * exp(-U * t / heat_capacity)
            T_steady = (U * weather_conditions.outdoor_temperature + 
                       hvac_power + additional_power_input) / U
            decay_rate = U / self.heat_capacity
            final_temp = T_steady + (initial_temperature - T_steady) * np.exp(-decay_rate * time_seconds)
            return final_temp


class DefaultBuildingModel(BuildingModel):
    """
    Default building model with typical residential construction.
    
    Provides a reasonable starting point for residential building simulations.
    All parameters are in SI units.
    """
    
    def __init__(self) -> None:
        """
        Initialize default building model with typical residential parameters.
        
        Parameters converted to SI units:
        - Studs: 2x4 studs (0.038m x 0.089m) at 16" (0.406m) spacing
        - Wall insulation: R-13 (2.29 m²·K/W)
        - Window: R-4 (0.70 m²·K/W) with 0.7 SHGC
        - Roof: R-60 (10.56 m²·K/W)
        - Floor: R-30 (5.28 m²·K/W)
        - Heat capacity: 100 kJ/K (typical for 100m² house)
        """
        # Convert typical US construction to SI units
        # Create default orientation (South-facing)
        default_orientation = Orientation()
        
        wall = WallModel(
            Studs(0.038, 0.089, 0.406),  # 2x4 studs at 16" spacing
            2.29,  # R-13 insulation in m²·K/W
            100,   # 100 m² wall area
            default_orientation
        )
        
        window = WindowModel(
            0.70,  # R-4 window in m²·K/W
            0,     # No windows in default model
            0.7,   # SHGC
            default_orientation
        )
        roof = RoofModel(
            10.56, # R-60 roof in m²·K/W
            0,     # No roof area in default model
            default_orientation,
            0.85   # Absorptivity
        )
        floor = PierAndBeam(
            Studs(0.038, 0.140, 0.406),  # 2x6 joists at 16" spacing
            5.28,  # R-30 floor in m²·K/W
            0,     # No floor area in default model
            default_orientation
        )
        heating_model = ElectricResistanceThermalDeviceModel()
        super().__init__([wall, window, roof, floor], heating_model, 10 * 6)  # 100 kJ/K heat capacity
