import numpy as np
from scipy.optimize import minimize
from BuildingModel import BuildingModel
from WeatherConditions import WeatherConditions, SolarIrradiation


class ModelPredictiveControl:
    def __init__(self, model, horizon, Q, R):
        self.model = model  # type: BuildingModel
        self.horizon = horizon  # Prediction horizon
        self.Q = Q  # State cost matrix
        self.R = R  # Control cost matrix (a single int for now with a single control input)
        self.u_prev = None  # Previous control input
        self.bounds = [-10000, 10000]
        self.step_size_s = 60

    def cost_function(self, u, x):
        cost = 0.0
        inside_temperature = x[0]
        solar_irradiation = x[1:4]
        outside_weather = x[4:7]
        set_point = x[7]
        for i in range(self.horizon):
            inside_temperature += self.model.temperature_change_per_s(
                inside_temperature,
                WeatherConditions(SolarIrradiation(*solar_irradiation), *outside_weather), u[i])
            cost += (set_point - inside_temperature) ** 2
            # cost += np.dot(np.dot(x_pred.T, self.Q), x_pred)
            cost += u[i] ** 2 * self.R

        # print(cost, u)
        return cost

    def control(self, x0):
        u0 = np.zeros((self.horizon, 1))  # Initial guess for control inputs

        # Define bounds for control inputs
        bounds = [self.bounds for _ in range(self.horizon)]

        # Define equality constraint for system dynamics
        # constraints = {'type': 'eq', 'fun': self.model.constraint}

        # Solve the optimization problem
        # result = minimize(self.cost_function, u0, args=(x0,), method='SLSQP',
        #                   bounds=bounds, constraints=constraints)

        result = minimize(self.cost_function, u0, args=(x0,), method='SLSQP',
                          bounds=bounds)
        # print(result)
        if result.success:
            u_opt = result.x[0]
        else:
            u_opt = self.u_prev

        self.u_prev = u_opt
        return u_opt
