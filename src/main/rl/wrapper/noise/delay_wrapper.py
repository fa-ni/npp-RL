import random

from gym import Wrapper


# TODO Test all Wrappers

# Needs to be implemented before the Obs Wrapper as the calculation of the observation should only take place after this
# wrapper
class DelayNoiseWrapperOption1(Wrapper):
    # Executed every 10th timestep
    def step(self, action):
        original_result = list(self.env.step(action))
        if self.length % 10 == 0:
            self.state.time_step(1)
        return original_result


class DelayNoiseWrapperOption2(Wrapper):
    # Executed with a 10 % chance // TODO Or use fibonacci?
    def step(self, action):
        original_result = list(self.env.step(action))
        if random.randint(0, 100) < 10:
            self.state.time_step(1)
        return original_result

class DelayNoiseWrapperOption3(Wrapper):
    # With every execution the risk increases by 5% // TODO Or use fibonacci?
    def __init__(self, env):
        super().__init__(env)
        self.current_risk = 0

    def step(self, action):
        original_result = list(self.env.step(action))
        if random.randint(0, 100) < self.current_risk:
            self.state.time_step(1)
            self.current_risk = 0
        else:
            self.current_risk += 5
        return original_result
