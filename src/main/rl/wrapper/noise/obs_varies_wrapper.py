import random

import numpy as np
from gym import Wrapper

# Further ideas:
"""
- Only set noise to 1 obs var
- Randomly set 1 obs var to a random value
- Apply noise to only some steps not all
- Use different levels of noise
- Is adding doing the same as deducting?
"""


def fibonacci_of(n):
    if n in {0, 1}:
        result = n
    else:
        result = fibonacci_of(n - 1) + fibonacci_of(n - 2)
    return result


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


class ObservationVariesNoiseWrapper(Wrapper):

    def step(self, action):
        original_result = list(self.env.step(action))
        new_obs = []
        for item in original_result:
            new_obs.append(item + 0.1 if item < 2 else 2)
        original_result[0] = np.array(new_obs)
        return original_result


# TODO Check if we could use also the fibonanci the other way around, so have a lot of noise in the beginning and later less
#  [250 - fibonacci_of(n) for n in range(14)]
class ObservationVariesNoiseWrapper1(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.fibonacci = [fibonacci_of(n) for n in range(14)]

    def step(self, action):
        original_result = list(self.env.step(action))
        new_obs = []

        if self.length in self.fibonacci:
            for item in original_result:
                new_obs.append(item + 0.1 if item < 2 else 2)
        original_result[0] = np.array(new_obs)
        return original_result


# TODO: Decide if random setting is really a good idea -> Comparison might be hard
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
