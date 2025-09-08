# MPC HVAC Controller

Model Predictive Control HVAC system with CO2 and ventilation management.

The goal of this is to have a more intelligent heating/cooling schedule that takes into account building parameters, occupant comfort (temperature and CO2), electricity pricing, and predicted weather patterns. It can implicitly address questions like the following:

- Should I turn off my thermostat when leaving
- How much money will HVAC cost me today
- How should I leverage time of use pricing effectively
- If my equipment is undersized, how should I precondition the space before an extreme weather event
- With an only heat/cool system how should I ensure I don't over condition the space (i.e. heat too much in the cold morning and then overheat during the day).
- At what point is it cheaper to run the fans at night for night time cooling compared to using AC
- Which heating source is currently the most effective to use (i.e. air source heat pump, ground source heat pump, resistance, gas etc)
- Should I use backup heat when it's too cold for a heat pump or should I preheat/allow the temperature to sag slightly


While this was designed with buildings in mind where ventilation is also important, it can also be used for hot tubs or water heating schedules (see examples). This is best suited where the thermal and/or ventilation loads are strongly deferable, so a fridge would not be a good candidate because temperatures should remain stable and the thermal mass is low. However, ideally ice making and defrosting would occur during cheaper electricity times.

At a high level, a traditional thermostat works as a [pid](https://en.wikipedia.org/wiki/PID_controller) controller by having a set point and changing a state variable (heat/cool) to reach that set point at each point in time. The set point will change at times with a set schedule, based on the state of people in the building, or can go to vacation mode. It doesn't do any planning and is reactive to external variables.

The goal of this is to use a [model predictive controller](https://en.wikipedia.org/wiki/Model_predictive_control) to define a better trajectory than a set schedule to further minimize energy cost while maintaining comfort. This also includes ventillation for controlling CO2 levels and/or capturing free cooling or heating.

The controller optimizes over a prediction horizon (typically 24 hours) to find the best trajectory that balances energy efficiency, comfort, and air quality. Once it implements the first step, it will then recalculate the entire next 24 hour horizon. This is the control loop, and 15-30 minutes between predictions is probably sufficient. 

The optimization objective looks like this

$$ \sum_i w_i e + \sum_i (t_i - t_{set}) ^ 2 + \sum_i (co_2 - co_{2set}) ^ 2 + \sum_i \int t_{error} + \sum_i \int co_{2error}$$

The first term is the energy use times the energy price, the second and third terms are the cost for violating the temperature and co2 set points, and the fourth and fifth terms are integrals on the temperature and co2 errors to ensure that at steady state the errors trend towards 0. These integral terms both reset after a point has 0 error, much like the integral term on a pid controller. Co2 and temperature costs are set to 0 when the space is not occupied. Temperature and co2 levels use penalties rather than constraints to allow not fully controllable systems to operate (i.e. heat only or undersized equipment), allow for trading comfort for cost, and ensure a trajectory is able to be found for a variety of desired schedules.

During each time step, the ventilation rates could be controlled by using a PWM signal to precisely control the fan speeds or by simply turning the ventilation on/off. 

For heating and cooling, the controls are a little less clear and I think the best option would be to set the thermostat to the specified mode (i.e. heat, cool, or off) and change the set point to the expected indoor temperate at that time step. 

All units are in SI. 

### Currently supported features
- Weekly schedules that include temperature set points and deadbands, co2 set points, electricity pricing, and occupancy for home and away planning
- Multiple ventilation types simultaneously
- Adjustable weights for co2, comfort, and electricity costs
- Visual representation of the planned trajectory
- Occupancy scheduling ensures that CO2 and heat production of occupants is only considered when present

### Currently planned to be supported
- Multiple simultaneous hvac systems (i.e. air source heat pump and electric resistance strips)
- Faster optimization

### Features under consideration
- Feels-like temperature that includes humidity levels
- Controlling and modeling humidity levels
- More accurate simulation on erv/hrv efficiencies vs fan speed
- More accurate simulation of heat pumps output capacities based on indoor and outdoor temperatures
- Solar heat gain on all surfaces

<h2>Assumptions</h2>

1. Cost of energy does not depend on the amount used (i.e. constant marginal rates for each time)
2. Power is not considered a factor
3. Different sources of energy are treated the same (locally generated solar and grid usage are fungible. This is implicitly assumed in (1))

At each time step, the temperature evolves according to

$$ T_{i + 1} = T_i + \frac{\alpha + \beta \delta_T + Q \delta_T c_p}{c_b}   $$

Where $alpha$ is the baseload heat generated inside the house (includes occupants), $\beta$ is the net effective r value of the building times the area, $\delta_T$ is the difference in temperature between inside and outside, Q is the airflow, $c_p$ is the heat capacity of air, and $c_b$ is the heat capacity of the building.

Similarly, CO2 evolves with 

$$C_{i + 1} = C_i + \frac{\sigma o + Q (C_i - C_o)}{V} $$

Where $\sigma$ is the co2 generation rate per occupant, o is the number of occupants, Q is the airflow, $C_o$ is the outdoor co2 level, and V is the room volume
<h2>Installation</h2>

<h3>Add-On in Home Assistant</h3>
This can be installed as an add-on using home assistant. It is registered to port 8000 and can be installed by simply using this repo as the add-on source. Then the webpage can be accessed via http://homeassistantip:8000.
<br><br>

**Please note**: optimizaion is currently quite slow. 1-2 minutes is quite possible with 3 controllable variables and 48 steps (0.5 hour step size and 24 hour horizon). 

<h3>Sending Requests to the Add-On</h3>
There are two changes needed to actually send data to the add-on. A REST command has to be registered in configuration.yaml like

```
rest_command:
  send_prediction_request:
    url: http://localhost:8000/predict
    method: POST
    content_type: application/json
    payload: "{{ prediction_payload }}"
    timeout: 300
```
And then set up an automation like this to periodically send requests (my HA instance is in Fahrenheit so I have to convert to degrees Celsius in this automation)

```
alias: Make MPC Prediction API Request
description: ""
triggers:
  - trigger: time_pattern
    hours: /1
    id: time
conditions: []
actions:
  - action: weather.get_forecasts
    metadata: {}
    data:
      type: hourly
    target:
      entity_id: weather.forecast_home
    response_variable: forecast_response
  
  - action: rest_command.send_prediction_request
    data:
      prediction_payload: >

        {% set forecasts =
        forecast_response["weather.forecast_home"]["forecast"] %} {
          "current_co2_ppm": {{ states("co2_sensor") | float }},
          "current_temp_c": {{ ((states("sensor.thermostat_temperature") | float + states("other temperature sensor") | float) / 2 - 32) * 5/9 }},
          "current_time": "{{ now().isoformat() }}",
          "current_humidity":  {{ states("sensor.awair_element_66681_humidity") | float / 100 }},
          "weather_time_series": [
            {% for f in forecasts %}
              {
                "time": "{{ f.datetime }}",
                "outdoor_temperature": {{ (f.temperature - 32) * 5/9 }},
                "wind_speed": {{ f.wind_speed }},
                "solar_altitude_rad": 0.0,
                "solar_azimuth_rad": 0.0,
                "solar_intensity_w": 0.0,
                "ground_temperature": 12.0,
                "humidity": {{ f.humidity / 100.0 }},
              }{% if not loop.last %},{% endif %}
            {% endfor %}
          ]
        }
      response_variable: response
```


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

For cooling I will use the same formula for the COP - 1 and ignore any latent cooling.

## Overview

### Running Examples

```bash
# specific use cases to get a sense of the benefits
# example of a hot tub with only a few hours of possible occupancy a day and variable electric pricing
python3 example_prediction_hot_tub.py

# building is only occupied a few hours a day
python3 example_periodically_occupied_building.py

# Main integrated HVAC example (can take ~ 10 min to complete)
python3 example_hvac_controller.py

# running the server locally
python3 start_server.py --host 0.0.0.0 --port 8000 --config-file ./data/hvac_config.json

#Then run
/tests/test_server/integration_test_prediction_api.py

#to send APIs to the server and see the best planned trajectory
```

## Configuration

### Building Parameters

The `DefaultBuildingModel` provides typical residential parameters:
- **Walls**: R-13 insulation (2.29 m²·K/W)
- **Windows**: R-4 (0.70 m²·K/W) with 0.7 SHGC
- **Roof**: R-60 (10.56 m²·K/W)
- **Floor**: R-30 (5.28 m²·K/W)
- **Heat capacity**: 100 kJ/K (not really sure if this is a good value or not)

### Controller Parameters

- **`horizon_hours`**: Prediction horizon (default: 24 hours)
- **`co2_weight`**: Weight for CO₂ deviation penalty
- **`energy_weight`**: Weight for energy costs
- **`comfort_weight`**: Weight for temperature comfort
- **`step_size_hours`**: Step sizes used for planning. Total points optimized would then be horizon_hours * step_size_hours 
