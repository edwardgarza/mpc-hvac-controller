import unittest
from src.models.humidity import humidity


class TestHumidityUtils(unittest.TestCase):

    def test_humimdity_inverses(self):
        for temp in range(0, 30, 5):
            for hum in range(0, 100, 10):
                abs_hum = humidity.absolute_humidity_from_relative(temp, hum / 100)
                self.assertAlmostEqual(humidity.relative_humidity_from_asbolute(temp, abs_hum), hum / 100)

    def test_humidity_values(self):
        self.assertAlmostEqual(humidity.absolute_humidity_from_relative(20, 0.5), 8.617, delta=0.001)