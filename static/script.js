// Navigation
document.addEventListener('DOMContentLoaded', function() {
    // Navigation handling
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Show target section
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });
        });
    });
    
    // Load initial status
    refreshStatus();
});

// API Functions
async function apiCall(method, endpoint, data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Dashboard Functions
async function refreshStatus() {
    try {
        const result = await apiCall('GET', '/current-control');
        result.JSON
        // For now, we'll use mock data since we don't have real sensor integration
        // In a real implementation, this would call your sensor endpoints
        document.getElementById('co2-level').textContent = result.co2_level.toFixed(1);
        document.getElementById('temp-level').textContent = result.indoor_temperature.toFixed(1);
        const ventString = Object.entries(result.ventilation_controls)
        .map(([type, rate]) => `${type}: ${rate.toFixed(0)} m³/hr`)
        .join('<br>');
        document.getElementById('ventilation-status').innerHTML = ventString;
        const hvacString = Object.entries(result.hvac_controls)
        .map(([type, rate]) => `${type}: ${(rate / 1000).toFixed(1)} kW`)
        .join('<br>');
    
        document.getElementById('hvac-status').innerHTML = hvacString;
        document.getElementById('estimated-cost').textContent = '$' + result.estimated_cost_over_horizon.toFixed(3);
        document.getElementById('unoptimized-cost').textContent = '$' + result.naive_cost_over_horizon_pid.toFixed(3);


    } catch (error) {
        console.error('Failed to refresh status:', error);
    }
}

async function runPrediction() {
    try {
        const weatherData = createSampleWeatherData();
        const requestData = {
            current_co2_ppm: 800.0,
            current_temp_c: 22.0,
            current_time_hours: 0.0,
            weather_data: weatherData,
            horizon_hours: 6.0
        };
        
        const result = await apiCall('POST', '/predict', requestData);
        console.log('Prediction result:', result);
        
        // Update status with prediction info
        if (result.has_prediction) {
            document.getElementById('ventilation-status').textContent = 'Optimized';
            document.getElementById('hvac-status').textContent = 'Optimized';
            
            // Display trajectory information
            if (result.co2_trajectory && result.temperature_trajectory) {
                const initialCo2 = result.co2_trajectory[0];
                const finalCo2 = result.co2_trajectory[result.co2_trajectory.length - 1];
                const initialTemp = result.temperature_trajectory[0];
                const finalTemp = result.temperature_trajectory[result.temperature_trajectory.length - 1];
                
                document.getElementById('co2-level').textContent = `${initialCo2.toFixed(0)} → ${finalCo2.toFixed(0)}`;
                document.getElementById('temp-level').textContent = `${initialTemp.toFixed(1)} → ${finalTemp.toFixed(1)}`;
                
                console.log(`CO2 trajectory: ${initialCo2.toFixed(1)} → ${finalCo2.toFixed(1)} ppm`);
                console.log(`Temperature trajectory: ${initialTemp.toFixed(1)} → ${finalTemp.toFixed(1)} °C`);
            }
        }
        
        alert('Prediction completed successfully! Check console for trajectory details.');
    } catch (error) {
        alert('Prediction failed: ' + error.message);
    }
}

async function generatePlot() {
    try {
        const plotContainer = document.getElementById('plot-container');
        plotContainer.innerHTML = '<p>Generating plot...</p>';
        
        const result = await apiCall('GET', '/plot-prediction');
        
        if (result.plot_data) {
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + result.plot_data;
            img.alt = 'Prediction Plot';
            plotContainer.innerHTML = '';
            plotContainer.appendChild(img);
        } else {
            plotContainer.innerHTML = '<p>No plot data received</p>';
        }
    } catch (error) {
        document.getElementById('plot-container').innerHTML = '<p>Error: ' + error.message + '</p>';
    }
}

// Configuration Functions
async function loadConfig() {
    try {
        const config = await apiCall('GET', '/config');
        const editor = document.getElementById('config-editor');
        editor.value = JSON.stringify(config, null, 2);
    } catch (error) {
        alert('Failed to load configuration: ' + error.message);
    }
}

async function saveConfig() {
    try {
        const editor = document.getElementById('config-editor');
        const configText = editor.value;
        
        // Validate JSON
        const config = JSON.parse(configText);
        
        // Save to server
        await apiCall('POST', '/config', config);
        alert('Configuration saved successfully!');
    } catch (error) {
        if (error instanceof SyntaxError) {
            alert('Invalid JSON format: ' + error.message);
        } else {
            alert('Failed to save configuration: ' + error.message);
        }
    }
}

function downloadConfig() {
    const editor = document.getElementById('config-editor');
    const configText = editor.value;
    
    if (!configText.trim()) {
        alert('No configuration to download. Load a configuration first.');
        return;
    }
    
    try {
        // Validate JSON
        JSON.parse(configText);
        
        // Create download link
        const blob = new Blob([configText], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        // TODO: this won't work in the HA add-on 
        a.download = './data/hvac_config.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        alert('Invalid JSON format: ' + error.message);
    }
}

// API Testing Functions
async function testEndpoint(method, endpoint) {
    try {
        const resultDiv = document.getElementById('api-result');
        resultDiv.textContent = `Testing ${method} ${endpoint}...`;
        
        let result;
        if (method === 'POST' && endpoint === '/predict') {
            const weatherData = createSampleWeatherData();
            const requestData = {
                current_co2_ppm: 800.0,
                current_temp_c: 22.0,
                current_time_hours: 0.0,
                weather_data: weatherData,
                horizon_hours: 6.0
            };
            result = await apiCall(method, endpoint, requestData);
        } else {
            result = await apiCall(method, endpoint);
        }
        
        resultDiv.textContent = `${method} ${endpoint} - Success:\n${JSON.stringify(result, null, 2)}`;
    } catch (error) {
        document.getElementById('api-result').textContent = `${method} ${endpoint} - Error: ${error.message}`;
    }
}

// Utility Functions
function createSampleWeatherData() {
    const weatherData = [];
    for (let hour = 0; hour <= 24; hour += 2) {
        weatherData.push({
            hour: hour,
            outdoor_temperature: 15.0 + 5.0 * (hour / 24.0),
            wind_speed: 5.0,
            solar_altitude_rad: 0.5,
            solar_azimuth_rad: 0.0,
            solar_intensity_w: 800.0,
            ground_temperature: 12.0
        });
    }
    return weatherData;
} 