#!/usr/bin/env python3
"""
Test calendar module
"""

import unittest
from src.utils.calendar import Calendar
from typing import Dict, List, Any
import datetime
import dateutil.parser

class TestCalendar(unittest.TestCase):

    def default_calendar(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"monday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15, "occupancy_count": 1},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25, "occupancy_count": 1},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10, "occupancy_count": 1}
        ],
        "tuesday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15, "occupancy_count": 1},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25, "occupancy_count": 1},
            {"time": "20:00", "co2": 900, "temperature": "20;24", "energy_cost": 0.10, "occupancy_count": 1}
        ],
        "wednesday": [
            {"time": "09:00", "co2": 800, "temperature": "20;24", "energy_cost": 0.15, "occupancy_count": 1},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25, "occupancy_count": 1},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10, "occupancy_count": 1}
        ],
        "thursday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15, "occupancy_count": 1},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25, "occupancy_count": 1},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10, "occupancy_count": 1}
        ],
        "friday": [
            {"time": "09:00", "co2": 800, "temperature": 22, "energy_cost": 0.15, "occupancy_count": 1},
            {"time": "17:00", "co2": 900, "temperature": 24, "energy_cost": 0.25, "occupancy_count": 1},
            {"time": "20:00", "co2": 900, "temperature": 22, "energy_cost": 0.10, "occupancy_count": 1}
        ],
        "saturday": [
            {"time": "10:00", "co2": 1000, "temperature": 20, "energy_cost": 0.08, "occupancy_count": 1}
        ],
        "sunday": [
            {"time": "12:00", "co2": 1000, "temperature": 20, "energy_cost": 0.08, "occupancy_count": 1}
        ]
        }


    def setUp(self):
        self.calendar = Calendar(
            self.default_calendar()
        )

    def test_get_relative_schedule(self):
        """Test that the relative schedule is correct"""
        # jan 1 2025 was a wednesday
        start_date = datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        r_s = self.calendar.get_relative_schedule(start_date)
        methods = [r_s.interpolate_step_co2, r_s.interpolate_step_temp, r_s.interpolate_step_energy_cost, r_s.interpolate_step_occupancy_count]
        self.assertSequenceEqual([x(0) for x in methods], [900, 22, 0.10, 1])

        print("deadband", r_s.interoplate_step_temp_deadband(0))
        # self.assertSequenceEqual(relative_schedule.interpolate_step(0), [900, 22, 0.10, 1])
        self.assertSequenceEqual([x(9) for x in methods], [800, 22, 0.15, 1])
        self.assertSequenceEqual([x(19) for x in methods], [900, 24, 0.25, 1])
        self.assertSequenceEqual([x(30) for x in methods], [900, 22, 0.1, 1])

    def test_get_relative_schedule_datetime_str_tzaware(self):
        """Test that the relative schedule is correct when considering time zones"""
        # Aug 12 2025 was a tuesday
        for tz in range(10):
            start_time_str =str.format("2025-08-12 12:00:00.00-0{0}:00", str(tz))
        # First, we need to call the predict endpoint to set up the weather series
            start_date = dateutil.parser.isoparse(start_time_str)
            r_s = self.calendar.get_relative_schedule(start_date)
            methods = [r_s.interpolate_step_co2, r_s.interpolate_step_temp, r_s.interpolate_step_energy_cost, r_s.interpolate_step_occupancy_count]

            self.assertSequenceEqual([x(0) for x in methods], [800, 22, 0.15, 1])
            self.assertSequenceEqual([x(5) for x in methods], [900, 24, 0.25, 1])
            self.assertSequenceEqual([x(8) for x in methods], [900, 22, 0.1, 1])
