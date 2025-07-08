"""
Factory for creating BuildingModel instances from configuration.
"""

from typing import List
from src.utils.config import BuildingConfig, WallConfig, WindowConfig, RoofConfig, FloorConfig, HeatingSystemConfig, CO2SourceConfig, RoomConfig
from src.models.building import (
    BuildingModel, WallModel, WindowModel, RoofModel, SlabModel, PierAndBeam,
    Studs
)
from src.models.heating import HeatingModel, ElectricResistanceHeatingModel, HeatPumpHeatingModel
from src.utils.orientation import Orientation
from src.controllers.ventilation.models import CO2Source


def create_orientation(orientation_float: float) -> Orientation:
    """Create Orientation instance from float value"""
    # Since Orientation is just a placeholder class, we just create an instance
    return Orientation()


def create_heating_model(config: HeatingSystemConfig) -> HeatingModel:
    """Create heating model from configuration"""
    output_range = (config.output_range_min, config.output_range_max)
    
    if config.type == "electric_resistance":
        return ElectricResistanceHeatingModel(output_range=output_range)
    elif config.type == "heat_pump":
        return HeatPumpHeatingModel(
            outdoor_offset=config.outdoor_offset,
            indoor_offset=config.indoor_offset,
            hspf=config.hspf,
            output_range=output_range
        )
    else:
        raise ValueError(f"Unknown heating system type: {config.type}")


def create_co2_sources(co2_source_configs: List[CO2SourceConfig]) -> List[CO2Source]:
    """Create CO2 sources from configuration"""
    co2_sources = []
    
    for source_config in co2_source_configs:
        # Create multiple sources if count > 1
        for _ in range(source_config.count):
            co2_source = CO2Source(
                co2_production_rate_m3_per_hour=source_config.co2_production_rate_m3_per_hour
            )
            co2_sources.append(co2_source)
    
    return co2_sources


def create_wall_models(wall_configs: List[WallConfig]) -> List[WallModel]:
    """Create wall models from configuration"""
    wall_models = []
    
    for wall_config in wall_configs:
        studs = Studs(
            width=wall_config.studs.width,
            depth=wall_config.studs.depth,
            spacing=wall_config.studs.spacing
        )
        
        wall_model = WallModel(
            studs=studs,
            insulation_r=wall_config.insulation_r,
            area_sq_m=wall_config.area_sq_m,
            orientation=create_orientation(wall_config.orientation),
            exterior_insulation_r=wall_config.exterior_insulation_r
        )
        wall_models.append(wall_model)
    
    return wall_models


def create_window_models(window_configs: List[WindowConfig]) -> List[WindowModel]:
    """Create window models from configuration"""
    window_models = []
    
    for window_config in window_configs:
        orientation = None
        if window_config.orientation is not None:
            orientation = create_orientation(window_config.orientation)
        
        window_model = WindowModel(
            insulation_r=window_config.insulation_r,
            area_sq_m=window_config.area_sq_m,
            solar_heat_gain_coefficient=window_config.solar_heat_gain_coefficient,
            orientation=orientation
        )
        window_models.append(window_model)
    
    return window_models


def create_roof_model(roof_config: RoofConfig) -> RoofModel:
    """Create roof model from configuration"""
    orientation = None
    if roof_config.orientation is not None:
        orientation = create_orientation(roof_config.orientation)
    
    return RoofModel(
        insulation_r=roof_config.insulation_r,
        area_sq_m=roof_config.area_sq_m,
        orientation=orientation or Orientation(),
        absorptivity=roof_config.absorptivity
    )


def create_floor_model(floor_config: FloorConfig) -> SlabModel | PierAndBeam:
    """Create floor model from configuration"""
    if floor_config.type == "slab":
        return SlabModel(
            insulation_r=floor_config.insulation_r,
            area_sq_m=floor_config.area_sq_m
        )
    elif floor_config.type == "pier_and_beam":
        studs = Studs(
            width=floor_config.studs.width,
            depth=floor_config.studs.depth,
            spacing=floor_config.studs.spacing
        )
        
        return PierAndBeam(
            studs=studs,
            insulation_r=floor_config.insulation_r,
            area_sq_m=floor_config.area_sq_m,
            orientation=Orientation(),  # Floor orientation doesn't matter for pier and beam
            exterior_insulation_r=floor_config.exterior_insulation_r
        )
    else:
        raise ValueError(f"Unknown floor type: {floor_config.type}")


def create_building_model(building_config: BuildingConfig) -> BuildingModel:
    """Create a complete BuildingModel from configuration"""
    
    # Create all thermal models
    thermal_models = []
    
    # Add walls
    thermal_models.extend(create_wall_models(building_config.walls))
    
    # Add windows
    thermal_models.extend(create_window_models(building_config.windows))
    
    # Add roof if specified
    if building_config.roof:
        thermal_models.append(create_roof_model(building_config.roof))
    
    # Add floor if specified
    if building_config.floor:
        thermal_models.append(create_floor_model(building_config.floor))
    
    # Create heating model
    heating_model = create_heating_model(building_config.heating_system)
    
    # Create and return the building model
    return BuildingModel(
        thermal_models=thermal_models,
        heating_model=heating_model,
        heat_capacity=building_config.heat_capacity
    )


def create_default_building_model() -> BuildingModel:
    """Create a default building model for testing"""
    from src.utils.config import config
    default_config = config.get_default_config()
    return create_building_model(default_config.building) 