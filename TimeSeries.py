from abc import ABC, abstractmethod
from typing import List
from scipy.interpolate import interp1d


class ArrayConversion(ABC):

    @classmethod
    @abstractmethod
    def from_array(cls, arr):
        pass

    @abstractmethod
    def to_array(self):
        pass


class TimeSeries:

    def __init__(self, ticks: List[float], values: List[ArrayConversion]):
        assert len(ticks) == len(values)
        self.ticks = ticks
        self.val_type = values[0]
        self.raw_values = values
        self.values = [x.to_array() for x in values]
        self.interp_func = interp1d(ticks, self.values, axis=0)

    def __getitem__(self, item):
        return self.raw_values[item]

    def interpolate(self, time):
        val = self.interp_func(time)
        return self.val_type.from_array(val)
