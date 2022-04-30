import gym as gym

from src.main.rl.utils.utils import is_done
from src.main.rl.wrapper.reward_calculations import calculate_roofed_reward


class RewardOption2Wrapper(gym.RewardWrapper):
    """
    This reward wrapper is used to exchange the reward function. It uses a "roofed" reward.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew: float) -> float:
        result = 0
        if not self.done:
            result = calculate_roofed_reward(self.state.full_reactor.generator.power)
        return result
