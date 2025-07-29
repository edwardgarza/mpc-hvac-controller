#!/usr/bin/env python3
"""
Test calendar module
"""

import unittest
from src.utils.calendar import Calendar
from typing import Dict, List, Any
from datetime import datetime, timedelta
class TestCalendar(unittest.TestCase):

    def default_calendar(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"monday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10}
        ],
        "tuesday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10}
        ],
        "wednesday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10}
        ],
        "thursday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10}
        ],
        "friday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10}
        ],
        "saturday": [
            {"time": "10:00", "co2": 1000, "temperature": 20, "energy_cost": 0.08}
        ],
        "sunday": [
            {"time": "12:00", "co2": 1000, "temperature": 20, "energy_cost": 0.08}
        ]
        }


    def setUp(self):
        self.calendar = Calendar(
            self.default_calendar()
        )

    def test_get_relative_schedule(self):
        """Test that the relative schedule is correct"""
        # jan 1 2025 was a wednesday
        start_date = datetime(2025, 1, 1, 0, 0, 0)
        relative_schedule = self.calendar.get_relative_schedule(start_date)
        self.assertSequenceEqual(relative_schedule.interpolate_step(0), [800, 22, 0.15])
        self.assertSequenceEqual(relative_schedule.interpolate_step(19), [900, 24, 0.25])
        self.assertSequenceEqual(relative_schedule.interpolate_step(30), [900, 22, 0.1])