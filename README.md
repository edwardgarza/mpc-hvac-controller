# MPC HVAC Controller

Model Predictive Control HVAC system with CO2 and ventilation management.

<!-- Updated to test GitHub Action -->
The goal of this is to have a more intelligent heating/cooling schedule that takes into account building parameters, occupant comfort, and predicted weather patterns.

At a high level, a traditional thermostat works as a [pid](https://en.wikipedia.org/wiki/PID_controller) controller by having a set point and changing a state variable (heat/cool) to reach that set point at each point in time. The set point will change at times with a set schedule, based on the state of people in the building, or can go to vacation mode. The goal of this is to use a [model predictive controller](https://en.wikipedia.org/wiki/Model_predictive_control) to define a better trajectory than a set schedule to further minimize energy use, cost, or even carbon intensity of the electric sources. This also includes ventillation for controlling CO2 levels.

Time of use pricing and scheduling for different temperatures or occupancy levels is currently not supported.

The controller optimizes over a prediction horizon (typically 24 hours) to find the best trajectory that balances energy efficiency, comfort, and air quality. All units are in SI. Once it implements the first step, it will then recalculate the entire next 24 hour horizon. 

### Controller Types

1. **`HvacController`** - **Main integrated controller**
   - Controls both heating/cooling AND ventilation
   - Optimizes all systems simultaneously so can capture free heating/cooling from open windows
   - Recommended for most use cases

2. **`IntegratedVentilationMpcController`** - **Ventilation-only controller**
   - Controls only ventilation systems
   - Useful for ventilation-only scenarios or debugging


<h2>Assumptions</h2>

1. Cost of energy does not depend on the amount used (i.e. constant marginal rates for each time)
2. Power is not considered a factor
3. Different sources of energy are treated the same (locally generated solar and grid usage are fungible. This is implicitly assumed in (1))

<h2>Building Modeling</h2>

Buildings will be modeled as easily as possible to begin, with increasing complexity depending on desired accuracy.

<h3>Energy Transfer</h3>
<h4>Conduction</h4>

Conduction will be one of the primary drivers in this simulation because it is both relatively easy to predict temperatures and because it is probably the main contriubtor to heat loss in colder weather. This simulation will use the following formula for heat flow

q = kA/s * $\delta$ t

where q is the power flow (J/s), k is the conductivity, A is the area, and s is the thickness of the material. R value is proportional to s/k.

<h4>Radiation</h4>

Radiation will be incorporated by using the [sol-air temperature](https://en.wikipedia.org/wiki/Sol-air_temperature) to replace the outdoor temperature **NOT IMPLEMENTED**

<h4>Convection</h4>

Initially convection will be ignored largely because I don't know how to model the amount of convection well for a given building as that depends on 

1. Mechanical ventillation systems
2. Building leak area
3. Outdoor temperature
4. Wind speed

I am hoping this will be relatively small and constant for areas and can be compensated for by the Natural Ventillation class.

<h3>Building Attributes</h3>

Buildings have windows, walls, roofs, and floors explicitly modeled. They will be modeled as simply as possible with as few objects and may not exactly map to reality. Once again, the hope here is that the model will be able to compensate with different parameters.

<h3>Heating/Cooling Modeling</h3>
Both electric resistance and combustion fuel have efficiencies independent of external temperature. However, heat pumps and air conditioner units have variable efficiencies.

From this link

https://www.nature.com/articles/s41597-019-0199-y

We can estimate COP of an air source heat pump to first order as 6.08 - 0.09 $\delta t$

For cooling I will use the same formula for the COP - 1.

## Overview

### Running Examples

```bash
# Main integrated HVAC example
python3 example_hvac_controller.py

# Ventilation-only example
python3 co2_control/integrated_ventilation_example.py
```

## Configuration

### Building Parameters

The `DefaultBuildingModel` provides typical residential parameters:
- **Walls**: R-13 insulation (2.29 m²·K/W)
- **Windows**: R-4 (0.70 m²·K/W) with 0.7 SHGC
- **Roof**: R-60 (10.56 m²·K/W)
- **Floor**: R-30 (5.28 m²·K/W)
- **Heat capacity**: 100 kJ/K

### Controller Parameters

- **`horizon_hours`**: Prediction horizon (default: 24 hours)
- **`co2_weight`**: Weight for CO₂ deviation penalty
- **`energy_weight`**: Weight for energy costs
- **`comfort_weight`**: Weight for temperature comfort
- **`co2_target_ppm`**: Target CO₂ concentration (default: 800 ppm)
- **`temp_target_c`**: Target indoor temperature (default: 22°C)