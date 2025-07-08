# HVAC Controller Add-on for Home Assistant

A Model Predictive Control (MPC) HVAC system with integrated CO₂ control and ventilation management.

## Features

- **Model Predictive Control**: Optimizes HVAC and ventilation based on predictions
- **CO₂ Management**: Controls indoor air quality through ventilation
- **Energy Optimization**: Balances comfort, energy cost, and air quality
- **RESTful API**: Easy integration with Home Assistant automations
- **Web Interface**: Built-in dashboard for monitoring and configuration

## Installation

1. Add this repository to your Home Assistant add-on store
2. Install the "HVAC Controller" add-on
3. Configure your building parameters and preferences
4. Start the add-on

## Configuration

### Controller Settings
- **Horizon Hours**: Prediction time horizon (1-168 hours)
- **CO₂ Weight**: Importance of CO₂ control vs other factors
- **Energy Weight**: Importance of energy cost optimization
- **Comfort Weight**: Importance of temperature comfort
- **CO₂ Target**: Target indoor CO₂ concentration (400-2000 ppm)
- **Temperature Target**: Target indoor temperature (16-28°C)
- **Step Size**: Time step for predictions (0.1-2.0 hours)

### Building Settings
- **Room Volume**: Indoor space volume in cubic meters
- **Outdoor CO₂**: Outdoor CO₂ concentration
- **Heat Capacity**: Building thermal mass

## API Endpoints

Once running, the add-on provides these REST endpoints:

- `GET /` - Web dashboard
- `GET /docs` - API documentation
- `POST /predict` - Get optimization predictions
- `GET /plot-prediction` - View prediction plots
- `GET /config` - Get current configuration
- `POST /config` - Update configuration

## Home Assistant Integration

Use these integrations in your `configuration.yaml`:

```yaml
# RESTful sensor for predictions
sensor:
  - platform: rest
    name: "HVAC Prediction"
    resource: http://localhost:8000/predict
    method: POST
    headers:
      Content-Type: application/json
    payload: '{"current_co2_ppm": 800, "current_temp_c": 22, "current_time_hours": 0, "weather_data": []}'

# RESTful command for triggering predictions
rest_command:
  hvac_predict:
    url: http://localhost:8000/predict
    method: POST
    headers:
      Content-Type: application/json
    payload: '{"current_co2_ppm": 800, "current_temp_c": 22, "current_time_hours": 0, "weather_data": []}'
```

## Automation Example

```yaml
automation:
  - alias: "HVAC Optimization"
    trigger:
      - platform: time_pattern
        minutes: "/30"  # Every 30 minutes
    action:
      - service: rest_command.hvac_predict
```

## Support

For issues and questions, please check the logs in the add-on configuration page. 