from typing import Dict, List, Any
from datetime import datetime, timedelta

class RelativeScheduleTimeSeries:
    def __init__(self, time_offsets_hours: List[float], values: List[List[float]]) -> None:
        self.keys = time_offsets_hours
        self.values = values

    def interpolate_step(self, time_offset_hours: float) -> List[float]:
        for i in reversed(range(len(self.keys))):
            if time_offset_hours >= self.keys[i]:
                return self.values[i]
        return self.values[0]

    def interpolate_step_co2(self, time_offset_hours: float) -> float:
        return self.interpolate_step(time_offset_hours)[0]

    def interpolate_step_temp(self, time_offset_hours: float) -> float:
        return self.interpolate_step(time_offset_hours)[1]
    
    def interpolate_step_energy_cost(self, time_offset_hours: float) -> float:
        return self.interpolate_step(time_offset_hours)[2]

    def interpolate_step_occupancy_count(self, time_offset_hours: float) -> float:
        return self.interpolate_step(time_offset_hours)[3]

class Calendar:
    def __init__(self, weekly_schedule: Dict[str, List[Dict[str, Any]]]):
        """
        Convert weekly schedule to absolute datetime entries
        
        Args:
            weekly_schedule: Weekly schedule dict with day_of_week keys
        """
        
        # Day of week mapping
        self.day_mapping = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        self.weekly_schedule = weekly_schedule

    def get_absolute_schedule(self, start_date_time: datetime, num_weeks: int = 2) -> List[List]:

        absolute_schedule = []

        current_date = start_date_time
        for week in range(num_weeks):
            for day_name, day_schedule in self.weekly_schedule.items():
                if day_name.lower() not in self.day_mapping:
                    continue
                    
                # Get the date for this day of the week
                days_ahead = self.day_mapping[day_name.lower()] - current_date.weekday() + 7 * week
                if days_ahead < 0:  # Target day already happened this week
                    continue
                day_date = current_date + timedelta(days=days_ahead)
                
                for entry in day_schedule:
                    # Create absolute datetime
                    absolute_time = f"{day_date.strftime('%Y-%m-%d')}T{entry['time']}:00Z"
                    
                    absolute_schedule.append(
                        [absolute_time, [entry.get("co2", 800.0), entry.get("temperature", 22.0), entry.get("energy_cost", 0.15), entry.get("occupancy_count", 1)]]
                    )
            
            # Move to next week
            current_date += timedelta(weeks=1)
        
        return absolute_schedule

    def get_relative_schedule(self, start_date_time: datetime, num_weeks: int = 2) -> RelativeScheduleTimeSeries:
        absolute_schedule = self.get_absolute_schedule(start_date_time, num_weeks)
        set_points = absolute_schedule
        return RelativeScheduleTimeSeries([(datetime.fromisoformat(x[0].replace('Z', '')) - start_date_time).total_seconds() / 3600.0 for x in set_points], [x[1] for x in set_points])
    
