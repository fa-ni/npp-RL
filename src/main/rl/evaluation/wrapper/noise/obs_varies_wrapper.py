import random

import numpy as np
from gym import Wrapper

from src.main.rl.utils.utils import fibonacci_of

obs_scaling_factors = {
    1: [800],
    3: [800, 2000, 100],
    7: [800, 2000, 100, 2000, 1, 1, 30],
    6: [800, 4000, 550, 8000, 180, 30],
    11: [800, 4000, 550, 8000, 180, 2000, 100, 2000, 1, 1, 30],
}


class ObservationVariesPositiveNoiseWrapper(Wrapper):
    def step(self, action):
        original_result = list(self.env.step(action))
        len_obs_space = len(original_result[0])
        current_obs_scaling_factors = obs_scaling_factors[len_obs_space]

        # Get real values and multiply with 1.05 for 5 %
        real_values_with_noise = [
            int(round((item + 1) * (current_obs_scaling_factors[idx] / 2))) * 1.05
            for idx, item in enumerate(original_result[0])
        ]
        # Normalize again
        new_obs = [2 * (item / current_obs_scaling_factors[idx]) - 1 for idx, item in enumerate(real_values_with_noise)]

        original_result[0] = np.array(new_obs)
        return original_result


class ObservationVariesNegativeNoiseWrapper(Wrapper):
    def step(self, action):
        original_result = list(self.env.step(action))
        len_obs_space = len(original_result[0])
        current_obs_scaling_factors = obs_scaling_factors[len_obs_space]
        # Get real values and multiply with 1.05 for 5 %
        real_values_with_noise = [
            int(round((item + 1) * (current_obs_scaling_factors[idx] / 2))) * 0.95
            for idx, item in enumerate(original_result[0])
        ]
        # Normalize again
        new_obs = [2 * (item / current_obs_scaling_factors[idx]) - 1 for idx, item in enumerate(real_values_with_noise)]
        original_result[0] = np.array(new_obs)
        return original_result


# TODO
class ObservationVariesNoiseWrapper1(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # TODO Check if we could use also the fibonacci the other way around
        self.fibonacci = [250 - fibonacci_of(n) for n in range(14)]
        self.maths_sign = "negative"

    def step(self, action):
        original_result = list(self.env.step(action))
        new_obs = []
        if self.length in self.fibonacci:
            len_obs_space = len(original_result[0])
            current_obs_scaling_factors = obs_scaling_factors[len_obs_space]
            # Get real values and multiply with 1.05 for 5 %
            if self.maths_sign == "negative":
                real_values_with_noise = [
                    int(round((item + 1) * (current_obs_scaling_factors[idx] / 2))) * 0.95
                    for idx, item in enumerate(original_result[0])
                ]
                self.maths_sign = "positive"
            else:
                real_values_with_noise = [
                    int(round((item + 1) * (current_obs_scaling_factors[idx] / 2))) * 1.05
                    for idx, item in enumerate(original_result[0])
                ]
            # Normalize again
            new_obs = [
                2 * (item / current_obs_scaling_factors[idx]) - 1 for idx, item in enumerate(real_values_with_noise)
            ]

            original_result[0] = np.array(new_obs)
        return original_result
