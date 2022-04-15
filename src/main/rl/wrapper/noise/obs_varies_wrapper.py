import random

import numpy as np
from gym import Wrapper

from src.main.rl.utils.utils import fibonacci_of


class ObservationVariesPositiveNoiseWrapper(Wrapper):
    def step(self, action):
        original_result = list(self.env.step(action))
        # for 5 % deviation in the normal value we need to add/subtract 0.1 from the scaled value
        new_obs = []
        for item in original_result[0]:
            new_obs.append(item + 0.1 if item < 2 else 2)
        original_result[0] = np.array(new_obs)
        return original_result


class ObservationVariesNegativeNoiseWrapper(Wrapper):
    def step(self, action):
        original_result = list(self.env.step(action))
        # for 5 % deviation in the normal value we need to add/subtract 0.1 from the scaled value
        new_obs = []
        for item in original_result[0]:
            new_obs.append(item - 0.1 if item > 0 else 0)
        original_result[0] = np.array(new_obs)
        return original_result


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
            for item in original_result:
                if self.maths_sign == "negative":
                    new_obs.append(item - 0.1 if item > 0 else 0)
                    self.maths_sign = "positive"
                else:
                    new_obs.append(item + 0.1 if item < 2 else 2)
        original_result[0] = np.array(new_obs)
        return original_result


class ObservationVariesNoiseWrapper2(Wrapper):
    # Set deviation between -5 % and +5 % randomly per step for all observations
    def step(self, action):
        original_result = list(self.env.step(action))
        new_obs = []
        # for 5 % deviation in the normal value we need to add/subtract 0.1 from the scaled value
        factor = random.randint(-10, 10) / 100
        for item in original_result:
            new_obs.append(item + factor if item < 2 else 2)
        original_result[0] = np.array(new_obs)
        return original_result
