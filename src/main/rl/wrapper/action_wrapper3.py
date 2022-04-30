import numpy as np
from gym import Wrapper
from gym.spaces import MultiBinary, Box, MultiDiscrete

from src.main.dto.FullReactor import FullReactor


class ActionSpaceOption3Wrapper(Wrapper):
    """
    Wrapper used to have 5 dimensions in the action space.
    """

    def __init__(self, env):
        super().__init__(env)
        # 1. CR/Moderator Percent 2. WP1 RPM 3. WV1 4. SV1 5. CP RPM
        if type(env.action_space) == MultiBinary:
            self.action_space = MultiBinary(5)
        elif type(env.action_space) == Box:
            self.action_space = Box(
                np.array([-1, -1, -1, -1, -1]).astype(np.float32), np.array([1, 1, 1, 1, 1]).astype(np.float32)
            )
        elif type(env.action_space) == MultiDiscrete:
            self.action_space = MultiDiscrete([9, 9, 2, 2, 9])

    def reset(self):
        self.env.reset()
        # override default values from scenarios back to standard start values
        # if there is no specific starting state
        if not self.starting_state:
            self.state.full_reactor.water_valve1.status = False
            self.state.full_reactor.steam_valve1.status = False
            self.state.full_reactor.condenser_pump.rpm = 0
        return_values = get_return_values_for_starting_state(self.state.full_reactor)
        return return_values


def get_return_values_for_starting_state(full_reactor: FullReactor):
    normalized_power = 2 * (full_reactor.generator.power / 800) - 1
    return np.array([float(normalized_power)])
