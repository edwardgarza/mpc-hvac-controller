#!/usr/bin/env python3
"""
Configuration management for HVAC Controller API - Config Files Only
"""

import json
from pathlib import Path
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator


class StudsConfig(BaseModel):
    """Configuration for wall studs"""
    width: float = Field(default=0.038, ge=0.01, le=0.1, description="Width of stud in meters")
    depth: float = Field(default=0.089, ge=0.01, le=0.2, description="Depth of stud in meters")
    spacing: float = Field(default=0.406, ge=0.2, le=0.6, description="Center-to-center spacing in meters")


class WallConfig(BaseModel):
    """Configuration for a wall"""
    insulation_r: float = Field(default=2.0, ge=1.0, le=50.0, description="R-value of insulation in m²·K/W")
    area_sq_m: float = Field(default=100.0, ge=1.0, le=1000.0, description="Wall area in square meters")
    orientation: float = Field(default=0.0, description="Wall orientation (radians or degrees - TBD)")
    studs: StudsConfig = Field(default_factory=StudsConfig, description="Stud configuration")
    exterior_insulation_r: float = Field(default=0.0, ge=0.0, description="Exterior insulation R-value in m²·K/W (purely additive)")


class WindowConfig(BaseModel):
    """Configuration for windows"""
    insulation_r: float = Field(default=1.0, ge=0.1, le=10.0, description="R-value in m²·K/W")
    area_sq_m: float = Field(default=20.0, ge=0.0, le=100.0, description="Window area in square meters")
    solar_heat_gain_coefficient: float = Field(default=0.7, ge=0.0, le=1.0, description="Solar heat gain coefficient")
    orientation: Optional[float] = Field(default=None, description="Window orientation (radians or degrees - TBD)")


class RoofConfig(BaseModel):
    """Configuration for roof"""
    insulation_r: float = Field(default=10.0, ge=10.0, le=100.0, description="R-value in m²·K/W")
    area_sq_m: float = Field(default=50.0, ge=1.0, le=500.0, description="Roof area in square meters")
    absorptivity: float = Field(default=0.85, ge=0.1, le=1.0, description="Solar absorptivity")
    orientation: Optional[float] = Field(default=None, description="Roof orientation (radians or degrees - TBD)")


class FloorConfig(BaseModel):
    """Configuration for floor"""
    type: str = Field(default="pier_and_beam", description="Floor type (slab, pier_and_beam)")
    insulation_r: float = Field(default=6.0, ge=1.0, le=100.0, description="R-value in m²·K/W")
    area_sq_m: float = Field(default=50.0, ge=1.0, le=500.0, description="Floor area in square meters")
    studs: StudsConfig = Field(default_factory=StudsConfig, description="Stud configuration for pier and beam")
    exterior_insulation_r: float = Field(default=0.0, ge=0.0, description="Exterior insulation R-value in m²·K/W (purely additive)")
    
    @field_validator('type')
    @classmethod
    def validate_floor_type(cls, v):
        valid_types = ['slab', 'pier_and_beam']
        if v not in valid_types:
            raise ValueError(f'type must be one of {valid_types}')
        return v


class HeatingSystemConfig(BaseModel):
    """Configuration for heating/cooling system"""
    type: str = Field(default="heat_pump", description="Heating system type (heat_pump, electric_resistance)")
    output_range_min: float = Field(default=-10000.0, description="Minimum output power in watts")
    output_range_max: float = Field(default=10000.0, description="Maximum output power in watts")
    
    # Heat pump specific parameters
    hspf: float = Field(default=9.0, ge=5.0, le=15.0, description="HSPF rating for heat pump")
    outdoor_offset: float = Field(default=3.0, description="Outdoor temperature offset")
    indoor_offset: float = Field(default=5.0, description="Indoor temperature offset")
    
    @field_validator('type')
    @classmethod
    def validate_heating_type(cls, v):
        valid_types = ['heat_pump', 'electric_resistance']
        if v not in valid_types:
            raise ValueError(f'type must be one of {valid_types}')
        return v


class WindowVentilationConfig(BaseModel):
    """Configuration for window ventilation"""
    max_airflow_m3_per_hour: float = Field(default=300.0, ge=0.0, description="Max airflow in m³/h")
    fan_power_w_m3_per_hour: float = Field(default=0.0, ge=0.0, description="Fan power per m³/h (0 for passive)")

class ERVConfig(BaseModel):
    """Configuration for ERV (Energy Recovery Ventilator)"""
    max_airflow_m3_per_hour: float = Field(default=200.0, ge=0.0, description="Max airflow in m³/h")
    heat_recovery_efficiency: float = Field(default=0.7, ge=0.0, le=1.0, description="Heat recovery efficiency (0-1)")
    fan_power_w_m3_per_hour: float = Field(default=0.3, ge=0.0, description="Fan power per m³/h")

class NaturalVentilationConfig(BaseModel):
    """Configuration for natural ventilation/infiltration"""
    infiltration_rate_ach: float = Field(default=0.2, ge=0.0, description="Infiltration rate (air changes per hour)")
    indoor_volume_m3: float = Field(default=100.0, ge=1.0, description="Indoor volume in m³")

class VentilationConfig(BaseModel):
    """Configuration for all ventilation types"""
    window_ventilations: List[WindowVentilationConfig] = Field(default_factory=lambda: [WindowVentilationConfig()])
    ervs: List[ERVConfig] = Field(default_factory=lambda: [ERVConfig()])
    natural_ventilations: List[NaturalVentilationConfig] = Field(default_factory=lambda: [NaturalVentilationConfig()])

class CO2SourceConfig(BaseModel):
    """Configuration for CO2 sources (occupants, pets, equipment, etc.)"""
    type: str = Field(default="occupant", description="Source type (occupant, pet, equipment, etc.)")
    co2_production_rate_m3_per_hour: float = Field(default=0.01, ge=0.0, description="CO2 production rate in m³/hour")
    count: int = Field(default=1, ge=0, description="Number of sources of this type")
    
    @field_validator('type')
    @classmethod
    def validate_source_type(cls, v):
        valid_types = ['occupant', 'pet', 'equipment', 'other']
        if v not in valid_types:
            raise ValueError(f'type must be one of {valid_types}')
        return v

class RoomConfig(BaseModel):
    """Configuration for room/building interior"""
    volume_m3: float = Field(default=100.0, ge=1.0, le=10000.0, description="Indoor volume in m³")
    outdoor_co2_ppm: float = Field(default=400.0, ge=300.0, le=1000.0, description="Outdoor CO2 concentration in ppm")
    co2_sources: List[CO2SourceConfig] = Field(default_factory=lambda: [CO2SourceConfig()], description="CO2 sources in the room")

class BuildingConfig(BaseModel):
    """Configuration for the entire building"""
    heat_capacity: float = Field(default=1000000.0, ge=100000.0, le=10 ** 10, description="Building heat capacity in J/K")
    baseload_interior_heating: float = Field(default=500.0, ge=0, description="Building baseload heating from lights, appliances, etc in W")
    room: RoomConfig = Field(default_factory=RoomConfig, description="Room configuration")
    walls: List[WallConfig] = Field(default_factory=list, description="Wall configurations")
    windows: List[WindowConfig] = Field(default_factory=list, description="Window configurations")
    roof: Optional[RoofConfig] = Field(default_factory=RoofConfig, description="Roof configuration")
    floor: Optional[FloorConfig] = Field(default_factory=FloorConfig, description="Floor configuration")
    heating_system: HeatingSystemConfig = Field(default_factory=HeatingSystemConfig, description="Heating system configuration")
    ventilation: VentilationConfig = Field(default_factory=VentilationConfig, description="Ventilation configuration")


class ControllerConfig(BaseModel):
    """HVAC Controller configuration with validation"""
    
    horizon_hours: float = Field(default=24.0, ge=1.0, le=168.0, description="Prediction horizon in hours")
    co2_weight: float = Field(default=1.0, ge=0.0, description="Weight for CO2 deviation penalty")
    energy_weight: float = Field(default=1.0, ge=0.0, description="Weight for energy cost")
    comfort_weight: float = Field(default=1.0, ge=0.0, description="Weight for temperature comfort")
    step_size_hours: float = Field(default=0.25, ge=0.1, le=2.0, description="Time step size")
    optimization_method: str = Field(default="SLSQP", description="Optimization method")
    max_iterations: int = Field(default=500, ge=10, le=2000, description="Maximum optimization iterations")
    
    @field_validator('optimization_method')
    @classmethod
    def validate_optimization_method(cls, v):
        valid_methods = ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr']
        if v not in valid_methods:
            raise ValueError(f'optimization_method must be one of {valid_methods}')
        return v


class ServerConfig(BaseModel):
    """Server configuration"""
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    reload: bool = Field(default=True, description="Enable auto-reload")
    log_level: str = Field(default="info", description="Log level")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v


class FullConfig(BaseModel):
    """Complete configuration including building, controller, server, and schedules"""
    controller: ControllerConfig = Field(default_factory=ControllerConfig, description="Controller configuration")
    building: BuildingConfig = Field(default_factory=BuildingConfig, description="Building configuration")
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    schedules: Optional[dict] = Field(default=None, description="Weekly schedule configuration")
    
    class Config:
        extra = "allow"  # Allow extra fields in JSON


class Config:
    """Main configuration class - Config Files Only"""
    
    def __init__(self):
        self.full_config: Optional[FullConfig] = None
        self.controller_config: Optional[ControllerConfig] = None
        self.building_config: Optional[BuildingConfig] = None
        self.server_config: Optional[ServerConfig] = None
        self.config_file_path: Optional[Path] = None
    
    def load_from_file(self, file_path: str) -> FullConfig:
        """Load configuration from JSON file"""
        config_path = Path(file_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            print(f"DEBUG: Loaded JSON keys: {list(config_dict.keys())}")
            if 'schedules' in config_dict:
                print(f"DEBUG: Schedules in JSON: {config_dict['schedules']}")
            else:
                print("DEBUG: No 'schedules' key in JSON")
            
            full_config = FullConfig(**config_dict)
            print(f"DEBUG: Parsed config has schedules: {hasattr(full_config, 'schedules')}")
            print(f"DEBUG: Parsed config.schedules: {getattr(full_config, 'schedules', None)}")
            
            return full_config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    def save_to_file(self, config: FullConfig, file_path: str):
        """Save configuration to JSON file"""
        config_path = Path(file_path)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config.model_dump(), f, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration file: {e}")
    
    def get_default_config(self) -> FullConfig:
        """Get default configuration with example building"""
        # Create a default building configuration
        default_building = BuildingConfig(
            room=RoomConfig(volume_m3=100.0, outdoor_co2_ppm=400.0),
            walls=[
                WallConfig(insulation_r=2.0, area_sq_m=100.0, orientation=0.0),
                WallConfig(insulation_r=2.0, area_sq_m=100.0, orientation=0.0),
                WallConfig(insulation_r=2.0, area_sq_m=50.0, orientation=0.0),
                WallConfig(insulation_r=2.0, area_sq_m=50.0, orientation=0.0)
            ],
            windows=[WindowConfig(insulation_r=0.8, area_sq_m=20.0, orientation=0.0)],
            roof=RoofConfig(insulation_r=10.0, area_sq_m=50.0, orientation=0.0),
            floor=FloorConfig(insulation_r=5.0, area_sq_m=50.0),
            heating_system=HeatingSystemConfig()
        )
        
        return FullConfig(
            controller=ControllerConfig(),
            building=default_building,
            server=ServerConfig(),
            schedules=None  # Default to no schedules
        )
    
    def load_config(self, config_file: Optional[str] = None) -> FullConfig:
        """Load configuration from file or use defaults"""
        
        # Try to load from file if specified
        if config_file:
            try:
                self.full_config = self.load_from_file(config_file)
                self.config_file_path = Path(config_file)
                print(f"Loaded configuration from file: {config_file}")
                return self.full_config
            except Exception as e:
                print(f"Warning: Could not load config from file {config_file}: {e}")
                print("Falling back to default configuration")
        
        # Use defaults only if no file was specified or file loading failed
        if self.full_config is None:
            self.full_config = self.get_default_config()
            print("Using default configuration")
        
        return self.full_config


# Global config instance
config = Config()


def get_controller_config(path: str) -> ControllerConfig:
    """Get the current controller configuration"""
    if config.full_config is None:
        config.load_config(path)
    return config.full_config.controller


def get_building_config(path: str) -> BuildingConfig:
    """Get the current building configuration"""
    if config.full_config is None:
        config.load_config(path)
    return config.full_config.building



def get_server_config() -> ServerConfig:
    """Get the current server configuration"""
    if config.server_config is None:
        config.server_config = ServerConfig()
    return config.server_config


def create_default_config_file(file_path: str = "hvac_config.json"):
    """Create a default configuration file"""
    try:
        config.save_to_file(config.get_default_config(), file_path)
        print(f"Created default configuration file: {file_path}")
    except Exception as e:
        print(f"Error creating config file: {e}")


if __name__ == "__main__":
    # Example usage
    print("HVAC Controller Configuration - Config Files Only")
    print("=" * 50)
    
    # Create default config file
    create_default_config_file()
    
    # Load config
    config_path = "./data/hvac_config.json"
    full_config = config.load_config(config_path)
    print(f"Loaded config: {full_config.model_dump()}")
    
    print("\nTo use a custom configuration:")
    print("1. Edit the config file with your preferred settings")
    print("2. Start the server with: python start_server.py --config-file ./data/hvac_config.json")
    
    print("\nExample configuration file:")
    example_config = {
        "controller": {
            "horizon_hours": 24.0,
            "co2_weight": 1.0,
            "energy_weight": 2.0,
            "comfort_weight": 1.5,
            "step_size_hours": 0.25,
            "optimization_method": "SLSQP",
            "max_iterations": 500,
        },
        "building": {
            "heat_capacity": 1000000.0,
            "room": {
                "volume_m3": 100.0,
                "outdoor_co2_ppm": 400.0
            },
            "walls": [
                {
                    "insulation_r": 13.0,
                    "area_sq_m": 100.0,
                    "orientation": 0.0
                }
            ],
            "windows": [
                {
                    "insulation_r": 4.0,
                    "area_sq_m": 20.0,
                    "solar_heat_gain_coefficient": 0.7,
                    "orientation": 0.0
                }
            ],
            "roof": {
                "insulation_r": 60.0,
                "area_sq_m": 50.0,
                "absorptivity": 0.85,
                "orientation": 0.0
            },
            "floor": {
                "type": "pier_and_beam",
                "insulation_r": 30.0,
                "area_sq_m": 50.0
            },
            "heating_system": {
                "type": "heat_pump",
                "hspf": 9.0,
                "output_range_min": -10000.0,
                "output_range_max": 10000.0
            }
        }
    }
    print(json.dumps(example_config, indent=2)) 