from typing import List

import numpy as np
from scipy.optimize import minimize
from BuildingModel import BuildingModel
from TimeSeries import TimeSeries
from WeatherConditions import WeatherConditions, SolarIrradiation


class ModelPredictiveControl:
    def __init__(self, model, horizon_h, Q, R):
        self.model = model  # type: BuildingModel
        self.horizon = horizon_h  # Prediction horizon in hours
        self.Q = Q  # State cost matrix
        self.R = R  # Control cost matrix (a single int for now with a single control input)
        self.u_prev = None  # Previous control input
        self.bounds = [-10000, 10000]
        self.step_size_h = 0.25

    def cost_function(self, u, inside_temperature: float, outside_weather: List[float], set_point: float):
        cost = 0.0
        for i in range(self.horizon):
            # print(i, u, outside_weather)

            inside_temperature += self.model.temperature_change_per_s(
                inside_temperature,
                WeatherConditions.from_array(outside_weather),
                u[i])
            cost += (set_point - inside_temperature) ** 2
            # cost += np.dot(np.dot(x_pred.T, self.Q), x_pred)
            cost += u[i] ** 2 * self.R

        # print(cost, u)
        return cost

    def control(self, inside_temperature: float, outside_weather: List[float], set_point: float):
        u0 = np.zeros((self.horizon, 1))  # Initial guess for control inputs

        # Define bounds for control inputs
        bounds = [self.bounds for _ in range(self.horizon)]

        # Define equality constraint for system dynamics
        # constraints = {'type': 'eq', 'fun': self.model.constraint}

        # Solve the optimization problem
        # result = minimize(self.cost_function, u0, args=(x0,), method='SLSQP',
        #                   bounds=bounds, constraints=constraints)
        result = minimize(self.cost_function,
                          u0,
                          args=(inside_temperature, outside_weather, set_point),
                          method='SLSQP',
                          bounds=bounds)
        # print(result)
        if result.success:
            u_opt = result.x[0]
        else:
            u_opt = self.u_prev

        self.u_prev = u_opt
        return u_opt
