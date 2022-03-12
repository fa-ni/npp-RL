import numpy as np
from gym import Wrapper

from src.main.rl.utils.utils import is_done
from src.main.services.NPPAutomationStepService import NPPAutomationStepService


class NPPAutomationWrapper(Wrapper):
    def step(self, action):
        self.env.step(action)
        reward = 0
        self.npp_automation.run()
        done = is_done(self.state.full_reactor, self.length)
        if not done:
            calc_reward = self.state.full_reactor.generator.power / 700
            reward += calc_reward
        normalized_obs = 2 * (self.state.full_reactor.generator.power / 800) - 1
        return [
            # Might need to change if we dont want to have binary for first observation
            np.array([normalized_obs]),
            reward,
            done,
            {},
        ]

    def reset(self):
        self.env.reset()
        self.npp_automation = NPPAutomationStepService(self.state)
        return np.array([float(-1)])
