from gym import Wrapper

# Needs to be implemented before the Obs Wrapper as the calculation of the observation should only take place after this
# wrapper
from src.main.rl.utils.utils import fibonacci_of


class DelayNoiseWrapperOption1(Wrapper):
    # Executed every 10th timestep
    def step(self, action):
        original_result = list(self.env.step(action))
        if self.length % 10 == 0:
            self.state.time_step(1)
        return original_result


class DelayNoiseWrapperOption2(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.fibonacci = [250 - fibonacci_of(n) for n in range(14)]

    def step(self, action):
        original_result = list(self.env.step(action))
        if self.length in self.fibonacci:
            self.state.time_step(1)

        return original_result
