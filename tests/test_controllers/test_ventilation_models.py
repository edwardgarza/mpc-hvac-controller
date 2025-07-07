#!/usr/bin/env python3
"""
Comprehensive tests for ventilation models, focusing on CO₂ dynamics
"""

import unittest
import numpy as np
import math
from src.controllers.ventilation.models import (
    RoomCO2Dynamics, CO2Source, WindowVentilationModel, 
    HRVModel, ERVModel, NaturalVentilationModel
)


class TestCO2Source(unittest.TestCase):
    """Test CO2 source behavior"""
    
    def test_co2_production_rate(self):
        """Test CO2 production rate is consistent"""
        source = CO2Source(co2_production_rate_m3_per_hour=0.02)
        
        # Should return the same rate regardless of time
        self.assertEqual(source.co2_production_rate(0), 0.02)
        self.assertEqual(source.co2_production_rate(100), 0.02)
        self.assertEqual(source.co2_production_rate(), 0.02)
    
    def test_co2_production_rate_units(self):
        """Test CO2 production rate has correct units"""
        # 0.02 m³/hour is typical for one person
        source = CO2Source(co2_production_rate_m3_per_hour=0.02)
        
        # Convert to ppm/s for a 100 m³ room
        room_volume_m3 = 100.0
        production_ppm_per_s = (source.co2_production_rate() / room_volume_m3) * 1e6 / 3600
        
        # Should be around 0.0056 ppm/s for one person in 100 m³ room
        # 0.02 m³/hour = 0.02/3600 m³/s = 0.00000556 m³/s
        # ppm = (0.00000556 / 100) * 1e6 = 0.0556 ppm/s
        self.assertAlmostEqual(production_ppm_per_s, 0.0556, delta=0.001)


class TestVentilationModels(unittest.TestCase):
    """Test individual ventilation model behavior"""
    
    def test_window_ventilation_airflow(self):
        """Test window ventilation airflow is direct"""
        window = WindowVentilationModel()
        
        # Airflow should equal control input
        self.assertEqual(window.airflow_m3_per_hour(50.0), 50.0)
        self.assertEqual(window.airflow_m3_per_hour(0.0), 0.0)
        self.assertEqual(window.airflow_m3_per_hour(200.0), 200.0)
    
    def test_window_ventilation_fan_power(self):
        """Test window ventilation has no fan power"""
        window = WindowVentilationModel()
        
        # Natural ventilation should have zero fan power
        self.assertEqual(window.fan_power_w(50.0), 0.0)
        self.assertEqual(window.fan_power_w(0.0), 0.0)
        self.assertEqual(window.fan_power_w(200.0), 0.0)
    
    def test_hrv_ventilation_fan_power(self):
        """Test HRV has fan power proportional to airflow"""
        hrv = HRVModel(fan_power_w_m3_per_hour=2.0)
        
        # Fan power should be proportional to airflow
        self.assertEqual(hrv.fan_power_w(100.0), 200) 
        self.assertEqual(hrv.fan_power_w(50.0), 100)  
        self.assertEqual(hrv.fan_power_w(0.0), 0.0)
    
    def test_erv_ventilation_fan_power(self):
        """Test ERV has fan power proportional to airflow"""
        erv = ERVModel(fan_power_w_m3_per_hour=1.5)
        
        # Fan power should be proportional to airflow
        self.assertEqual(erv.fan_power_w(100.0), 150)  
        self.assertEqual(erv.fan_power_w(50.0), 75)  
        self.assertEqual(erv.fan_power_w(0.0), 0)
    
    def test_natural_ventilation_airflow(self):
        """Test natural ventilation airflow calculation"""
        natural = NaturalVentilationModel(indoor_volume_m3=100.0, infiltration_rate_ach=0.2)
        
        # Airflow should be volume * infiltration rate
        expected_airflow = 100.0 * 0.2  # 20 m³/hour
        self.assertEqual(natural.airflow_m3_per_hour(), expected_airflow)
        self.assertEqual(natural.infiltration_flow_rate_m3_per_hour, expected_airflow)
    
    def test_energy_load_calculation(self):
        """Test energy load calculation for different ventilation types"""
        indoor_temp = 22.0
        outdoor_temp = 10.0 
        
        # Window ventilation (no heat recovery)
        window = WindowVentilationModel()
        window_load = window.energy_load_kw(100.0, indoor_temp, outdoor_temp)
        
        # HRV with 70% heat recovery
        hrv = HRVModel(heat_recovery_efficiency=0.7)
        hrv_load = hrv.energy_load_kw(100.0, indoor_temp, outdoor_temp)
        
        # ERV with 80% heat recovery
        erv = ERVModel(heat_recovery_efficiency=0.8)
        erv_load = erv.energy_load_kw(100.0, indoor_temp, outdoor_temp)
        
        # When outdoor is cold, ventilation removes heat (negative load)
        # Window should have highest magnitude load (no recovery)
        # HRV should have lower magnitude load than window
        # ERV should have lowest magnitude load (highest recovery)
        self.assertGreater(abs(window_load), abs(hrv_load))
        self.assertGreater(abs(hrv_load), abs(erv_load))
        self.assertAlmostEqual(erv_load, window_load * (1 - erv.heat_recovery_efficiency) + erv.fan_power_w(100.0) / 1000, delta=0.01)
        self.assertAlmostEqual(hrv_load, window_load * (1 - hrv.heat_recovery_efficiency) + hrv.fan_power_w(100.0) / 1000, delta=0.01)
        # Window and HRV should be negative (removing heat)
        # ERV can be positive (net heat gain) if heat recovery exceeds remaining heat loss
        self.assertLess(window_load, 0.0)
        self.assertLess(hrv_load, 0.0)
        # ERV load can be positive or negative depending on efficiency and conditions


class TestRoomCO2Dynamics(unittest.TestCase):
    """Test CO2 dynamics in a room"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.room_volume_m3 = 100.0
        self.outdoor_co2_ppm = 400.0
        
        # Create CO2 sources
        self.co2_sources = [CO2Source(co2_production_rate_m3_per_hour=0.02)]
        
        # Create ventilation models
        self.window_vent = WindowVentilationModel()
        self.hrv_vent = HRVModel(heat_recovery_efficiency=0.7)
        self.natural_vent = NaturalVentilationModel(
            indoor_volume_m3=self.room_volume_m3, 
            infiltration_rate_ach=0.1
        )
        
        # Create room dynamics
        self.room_dynamics = RoomCO2Dynamics(
            volume_m3=self.room_volume_m3,
            sources=self.co2_sources,
            controllable_ventilations=[self.window_vent],
            natural_ventilations=[self.natural_vent],
            outdoor_co2_ppm=int(self.outdoor_co2_ppm)
        )
    
    def test_co2_change_per_s_units(self):
        """Test CO2 change rate has correct units (ppm/s)"""
        initial_co2_ppm = 800.0
        control_inputs = [50.0]  # 50 m³/hour window ventilation
        
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # Should be a reasonable rate (typically -1 to +1 ppm/s)
        self.assertGreater(co2_change_per_s, -10.0)
        self.assertLess(co2_change_per_s, 10.0)
    
    def test_co2_change_with_no_ventilation(self):
        """Test CO2 increases when no ventilation"""
        initial_co2_ppm = 400.0
        control_inputs = [0.0]  # No window ventilation
        
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # CO2 should increase (positive change rate)
        self.assertGreater(co2_change_per_s, 0.0)
        self.assertAlmostEqual(self.room_dynamics.co2_levels_in_t(initial_co2_ppm, control_inputs, 15 * 60), 400 + co2_change_per_s * 15 * 60, delta=1.0)
    
    def test_co2_change_with_high_ventilation(self):
        """Test CO2 decreases when ventilation is high"""
        initial_co2_ppm = 1000.0  # High CO2
        control_inputs = [200.0]  # High ventilation
        
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # CO2 should decrease (negative change rate)
        self.assertLess(co2_change_per_s, 0.0)
    
    def test_co2_equilibrium_calculation(self):
        """Test CO2 equilibrium level calculation"""
        # Calculate equilibrium analytically
        total_production = sum(s.co2_production_rate() for s in self.co2_sources)
        total_airflow = 50.0  # 50 m³/hour ventilation
        
        # Equilibrium: production = removal
        # production = total_production * 1e6 / volume
        # removal = airflow * (co2_equilibrium - outdoor_co2) / volume
        # At equilibrium: production = removal
        # total_production * 1e6 = airflow * (co2_equilibrium - outdoor_co2)
        # co2_equilibrium = outdoor_co2 + (total_production * 1e6) / airflow
        
        expected_equilibrium = self.outdoor_co2_ppm + (total_production * 1e6) / total_airflow
        
        # Test with room dynamics
        control_inputs = [total_airflow]
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            expected_equilibrium, control_inputs
        )
        
        # At equilibrium, change should be close to zero
        self.assertAlmostEqual(co2_change_per_s, 0.0, delta=0.1)
    
    def test_co2_trajectory_convergence(self):
        """Test CO2 trajectory converges to equilibrium"""
        initial_co2_ppm = 1000.0
        control_inputs = [50.0]  # Moderate ventilation
        time_step_seconds = 3600  # 1 hour
        
        # Simulate for several hours
        current_co2 = initial_co2_ppm
        co2_history = [current_co2]
        
        for hour in range(10):  # 10 hours
            co2_change_per_s = self.room_dynamics.co2_change_per_s(
                current_co2, control_inputs
            )
            co2_change = co2_change_per_s * time_step_seconds
            current_co2 += co2_change
            co2_history.append(current_co2)
        
        # CO2 should converge (not oscillate wildly)
        self.assertLess(abs(co2_history[-1] - co2_history[-2]), 10.0)
        
        # Final CO2 should be reasonable (between outdoor and initial)
        self.assertGreater(current_co2, self.outdoor_co2_ppm)
        self.assertLess(current_co2, initial_co2_ppm)
    
    def test_co2_levels_in_t_consistency(self):
        """Test co2_levels_in_t gives consistent results with co2_change_per_s"""
        initial_co2_ppm = 800.0
        control_inputs = [50.0]
        time_step_seconds = 3600  # 1 hour
        
        # Method 1: Use co2_levels_in_t
        final_co2_method1 = self.room_dynamics.co2_levels_in_t(
            initial_co2_ppm, control_inputs, time_step_seconds
        )
        
        # Method 2: Use co2_change_per_s and integrate
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        co2_change = co2_change_per_s * time_step_seconds
        final_co2_method2 = initial_co2_ppm + co2_change
        
        # Results should be similar (within 5%)
        self.assertAlmostEqual(final_co2_method1, final_co2_method2, delta=0.05 * initial_co2_ppm)
    
    def test_multiple_ventilation_systems(self):
        """Test CO2 dynamics with multiple ventilation systems"""
        # Create room with multiple ventilation systems
        room_dynamics = RoomCO2Dynamics(
            volume_m3=self.room_volume_m3,
            sources=self.co2_sources,
            controllable_ventilations=[self.window_vent, self.hrv_vent],
            natural_ventilations=[self.natural_vent],
            outdoor_co2_ppm=int(self.outdoor_co2_ppm)
        )
        
        initial_co2_ppm = 1000.0
        control_inputs = [30.0, 20.0]  # Window + HRV
        
        co2_change_per_s = room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # Should have negative change rate (CO2 decreasing)
        self.assertLess(co2_change_per_s, 0.0)
    
    def test_zero_ventilation_edge_case(self):
        """Test CO2 dynamics with zero ventilation"""
        initial_co2_ppm = 400.0
        control_inputs = [0.0]  # No ventilation
        
        # Test co2_levels_in_t with zero ventilation
        final_co2 = self.room_dynamics.co2_levels_in_t(
            initial_co2_ppm, control_inputs, 3600
        )
        
        # CO2 should increase due to sources
        self.assertGreater(final_co2, initial_co2_ppm)
        
        # Test co2_change_per_s with zero ventilation
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # Should be positive (increasing CO2)
        self.assertGreater(co2_change_per_s, 0.0)
    
    def test_high_co2_edge_case(self):
        """Test CO2 dynamics with very high initial CO2"""
        initial_co2_ppm = 5000.0  # Very high CO2
        control_inputs = [100.0]  # High ventilation
        
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # Should have large negative change rate
        self.assertLess(co2_change_per_s, -1.0)
    
    def test_physical_consistency(self):
        """Test physical consistency of CO2 dynamics"""
        # Test that CO2 can't go below outdoor level with ventilation
        initial_co2_ppm = self.outdoor_co2_ppm
        control_inputs = [100.0]  # High ventilation
        
        co2_change_per_s = self.room_dynamics.co2_change_per_s(
            initial_co2_ppm, control_inputs
        )
        
        # At outdoor CO2 level, should only increase due to sources
        self.assertGreater(co2_change_per_s, 0.0)
    
    def test_ventilation_efficiency_comparison(self):
        """Test that different ventilation types have different energy costs but same CO2 effect"""
        indoor_temp = 22.0
        outdoor_temp = 10.0
        ventilation_rate = 50.0
        
        # Test energy loads
        window_load = self.window_vent.energy_load_kw(ventilation_rate, indoor_temp, outdoor_temp)
        hrv_load = self.hrv_vent.energy_load_kw(ventilation_rate, indoor_temp, outdoor_temp)
        
        # Window should have higher energy load (no heat recovery)
        self.assertGreater(abs(window_load), abs(hrv_load))
        
        # But CO2 removal should be the same for same airflow
        initial_co2 = 800.0
        window_control = [ventilation_rate]
        hrv_control = [ventilation_rate]
        
        window_co2_change = self.room_dynamics.co2_change_per_s(initial_co2, window_control)
        
        # Create room dynamics with HRV instead of window
        hrv_room = RoomCO2Dynamics(
            volume_m3=self.room_volume_m3,
            sources=self.co2_sources,
            controllable_ventilations=[self.hrv_vent],
            natural_ventilations=[self.natural_vent],
            outdoor_co2_ppm=int(self.outdoor_co2_ppm)
        )
        
        hrv_co2_change = hrv_room.co2_change_per_s(initial_co2, hrv_control)
        
        # CO2 change should be similar (same airflow)
        self.assertAlmostEqual(window_co2_change, hrv_co2_change, delta=0.1)


class TestVentilationModelIntegration(unittest.TestCase):
    """Test integration between different ventilation models"""
    
    def test_energy_cost_calculation(self):
        """Test energy cost calculation for different ventilation types"""
        indoor_temp = 22.0
        outdoor_temp = 5.0  # Cold outside
        ventilation_rate = 100.0
        
        # Test different ventilation types
        window = WindowVentilationModel()
        hrv = HRVModel(heat_recovery_efficiency=0.7)
        erv = ERVModel(heat_recovery_efficiency=0.8, moisture_recovery_efficiency=0.5)
        
        # Calculate energy costs
        window_cost = window.energy_cost_per_s(ventilation_rate, indoor_temp, outdoor_temp)
        hrv_cost = hrv.energy_cost_per_s(ventilation_rate, indoor_temp, outdoor_temp)
        erv_cost = erv.energy_cost_per_s(ventilation_rate, indoor_temp, outdoor_temp)
        
        # Window should have highest cost (no recovery)
        # HRV should have lower cost than window
        # ERV should have lowest cost (highest recovery)
        self.assertGreater(window_cost, hrv_cost)
        self.assertGreater(hrv_cost, erv_cost)
        
        # All costs should be positive (heating needed)
        self.assertGreater(window_cost, 0.0)
        self.assertGreater(hrv_cost, 0.0)
        self.assertGreater(erv_cost, 0.0)
    
    def test_fan_power_calculation(self):
        """Test fan power calculation for different ventilation types"""
        ventilation_rate = 100.0
        
        window = WindowVentilationModel()
        hrv = HRVModel(fan_power_w_m3_per_hour=2.0)
        erv = ERVModel(fan_power_w_m3_per_hour=1.5)
        
        # Window should have no fan power
        self.assertEqual(window.fan_power_w(ventilation_rate), 0.0)
        
        # HRV and ERV should have fan power
        self.assertEqual(hrv.fan_power_w(ventilation_rate), 200) 
        self.assertEqual(erv.fan_power_w(ventilation_rate), 150) 


if __name__ == "__main__":
    unittest.main() 