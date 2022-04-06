import numpy as np
from gym import Wrapper
from gym.spaces import MultiBinary, Box, MultiDiscrete

from src.main.dto.FullReactor import FullReactor


class ActionSpaceOption2Wrapper(Wrapper):
    def __init__(self, env):
        super().__init__(
            env,
        )
        # Maybe we can check here which original value it had, if multibinary we go like this else.. TODO
        # 1. CR/Moderator Percent 2. WP1 RPM 3. WV1
        if type(env.action_space) == MultiBinary:
            self.action_space = MultiBinary(3)
        elif type(env.action_space) == Box:
            self.action_space = Box(np.array([-1, -1, -1]).astype(np.float32), np.array([1, 1, 1]).astype(np.float32))
        elif type(env.action_space) == MultiDiscrete:
            self.action_space = MultiDiscrete([9, 9, 2])

    def reset(self):
        # override default values from scenarios back to standard start values
        # if there is no specific starting state
        self.env.reset()
        if not self.starting_state:
            self.state.full_reactor.water_valve1.status = False
        return_values = get_return_values_for_starting_state(self.state.full_reactor)
        return return_values


def get_return_values_for_starting_state(full_reactor: FullReactor):
    normalized_power = 2 * (full_reactor.generator.power / 800) - 1
    return np.array([float(normalized_power)])
