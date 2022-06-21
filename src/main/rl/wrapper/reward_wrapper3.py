import gym as gym

from src.main.rl.utils.utils import is_done
from src.main.rl.wrapper.reward_calculations import calculate_roofed_reward, calculate_reward_for_corridor


class RewardOption3Wrapper(gym.RewardWrapper):
    """
    This reward wrapper is used to exchange the reward function. It uses a more complicated reward which
    also takes into account the criticality of different states.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew: float) -> float:
        result = 0
        done = is_done(self.state.full_reactor, self.length)  # Maybe also set it as a class variable? TODO
        if not done and 680 < self.state.full_reactor.generator.power < 720:
            result = 1
        return result