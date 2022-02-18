import gym as gym

from src.main.rl.wrapper.reward_calculations import calculate_roofed_reward


class RewardOption2Wrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew: float) -> float:
        return calculate_roofed_reward(self.state.full_reactor.generator.power)
