import numpy as np
from gym import Wrapper
from gym.spaces import Box

from src.main.dto.FullReactor import FullReactor


class ObservationOption2Wrapper(Wrapper):
    """
    Wrapper used to have 3 dimensions in the Observation Space.
    1. Power Output 2. WP1 RPM 3. CP/moderator Percentage
    """

    def __init__(self, env):
        super().__init__(env)
        # 1. Power Output 2. WP1 RPM 3. CP/moderator Percentage
        self.observation_space = Box(np.array([-1, -1, -1]).astype(np.float32), np.array([1, 1, 1]).astype(np.float32))

    def step(self, action):
        original_result = list(self.env.step(action))
        normalized_rpm = 2 * (self.state.full_reactor.water_pump1.rpm / 2000) - 1
        normalized_moderator_percent = 2 * (self.state.full_reactor.reactor.moderator_percent / 100) - 1
        original_result[0] = np.append(
            original_result[0], np.array([float(normalized_rpm), float(normalized_moderator_percent)])
        )
        return tuple(original_result)

    def reset(self):
        self.env.reset()
        return_values = get_return_values_for_starting_state(self.state.full_reactor)
        return return_values


def get_return_values_for_starting_state(full_reactor: FullReactor):
    normalized_power = 2 * (full_reactor.generator.power / 800) - 1
    normalized_wp1_rpm = 2 * (full_reactor.water_pump1.rpm / 2000) - 1
    normalized_cr = 2 * (full_reactor.reactor.moderator_percent / 100) - 1

    return np.array([float(normalized_power), float(normalized_wp1_rpm), float(normalized_cr)])
