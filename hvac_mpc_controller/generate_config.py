#!/usr/bin/env python3
"""
Generate HVAC config from Home Assistant add-on options
"""

import json
import os
from pathlib import Path

def generate_config_from_options():
    """Generate hvac_config.json from add-on options"""
    
    # Get options from environment (set by Home Assistant)
    options = {
        # Controller settings
        "horizon_hours": float(os.getenv("horizon_hours", "24")),
        "co2_weight": float(os.getenv("co2_weight", "1.0")),
        "energy_weight": float(os.getenv("energy_weight", "1.0")),
        "comfort_weight": float(os.getenv("comfort_weight", "1.0")),
        "step_size_hours": float(os.getenv("step_size_hours", "0.25")),
        "optimization_method": os.getenv("optimization_method", "SLSQP"),
        "max_iterations": int(os.getenv("max_iterations", "500")),
        
        # Building settings
        "room_volume_m3": float(os.getenv("room_volume_m3", "100.0")),
        "outdoor_co2_ppm": float(os.getenv("outdoor_co2_ppm", "400.0")),
        "heat_capacity": float(os.getenv("heat_capacity", "1000000.0")),
        
        # Server settings
        "host": os.getenv("host", "0.0.0.0"),
        "port": int(os.getenv("port", "8000")),
        "log_level": os.getenv("log_level", "info"),
    }
    
    # Create the config structure
    config = {
        "controller": {
            "horizon_hours": options["horizon_hours"],
            "co2_weight": options["co2_weight"],
            "energy_weight": options["energy_weight"],
            "comfort_weight": options["comfort_weight"],
            "step_size_hours": options["step_size_hours"],
            "optimization_method": options["optimization_method"],
            "max_iterations": options["max_iterations"],
        },
        "building": {
            "heat_capacity": options["heat_capacity"],
            "room": {
                "volume_m3": options["room_volume_m3"],
                "outdoor_co2_ppm": options["outdoor_co2_ppm"],
                "co2_sources": [
                    {
                        "type": "occupant",
                        "co2_production_rate_m3_per_hour": 0.01,
                        "count": 1
                    }
                ]
            },
            "walls": [
                {
                    "insulation_r": 13.0,
                    "area_sq_m": 100.0,
                    "orientation": 0.0,
                    "studs": {
                        "width": 0.038,
                        "depth": 0.089,
                        "spacing": 0.406
                    },
                    "exterior_insulation_r": 0.0
                },
                {
                    "insulation_r": 13.0,
                    "area_sq_m": 100.0,
                    "orientation": 0.0,
                    "studs": {
                        "width": 0.038,
                        "depth": 0.089,
                        "spacing": 0.406
                    },
                    "exterior_insulation_r": 0.0
                },
                {
                    "insulation_r": 13.0,
                    "area_sq_m": 50.0,
                    "orientation": 0.0,
                    "studs": {
                        "width": 0.038,
                        "depth": 0.089,
                        "spacing": 0.406
                    },
                    "exterior_insulation_r": 0.0
                },
                {
                    "insulation_r": 13.0,
                    "area_sq_m": 50.0,
                    "orientation": 0.0,
                    "studs": {
                        "width": 0.038,
                        "depth": 0.089,
                        "spacing": 0.406
                    },
                    "exterior_insulation_r": 0.0
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
                "area_sq_m": 50.0,
                "studs": {
                    "width": 0.038,
                    "depth": 0.089,
                    "spacing": 0.406
                },
                "exterior_insulation_r": 0.0
            },
            "heating_system": {
                "type": "heat_pump",
                "output_range_min": -10000.0,
                "output_range_max": 10000.0,
                "hspf": 9.0,
                "outdoor_offset": 3.0,
                "indoor_offset": 5.0
            },
            "ventilation": {
                "window_ventilations": [
                    {
                        "max_airflow_m3_per_hour": 300.0,
                        "fan_power_w_m3_per_hour": 0.0
                    }
                ],
                "ervs": [
                    {
                        "max_airflow_m3_per_hour": 200.0,
                        "heat_recovery_efficiency": 0.7,
                        "fan_power_w_m3_per_hour": 0.3
                    }
                ],
                "natural_ventilations": [
                    {
                        "infiltration_rate_ach": 0.2,
                        "indoor_volume_m3": 100.0
                    }
                ]
            }
        },
        "server": {
            "host": options["host"],
            "port": options["port"],
            "reload": False,
            "log_level": options["log_level"]
        }
    }
    
    # Write config file
    config_path = Path("hvac_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated config file: {config_path}")
    print(f"Server will run on {options['host']}:{options['port']}")

if __name__ == "__main__":
    generate_config_from_options() 