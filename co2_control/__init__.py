"""
CO2 Control Module

This module contains all the components needed for CO2-based ventilation control
using Model Predictive Control (MPC).
"""

from .VentilationModels import (
    BaseVentilationModel, 
    WindowVentilationModel, 
    HRVModel, 
    ERVModel,
    NaturalVentilationModel,
    RoomCO2Dynamics,
    CO2Source
)

__all__ = [
    'BaseVentilationModel',
    'WindowVentilationModel', 
    'HRVModel', 
    'ERVModel',
    'NaturalVentilationModel',
    'RoomCO2Dynamics',
    'CO2Source',
] 