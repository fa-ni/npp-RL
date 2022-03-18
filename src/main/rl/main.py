from src.main.rl.training import train_all_scenarios
from src.main.rl.utils.constants import ALL_SCENARIOS
from src.main.rl.wrapper.npp_automation_wrapper import NPPAutomationWrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper
from stable_baselines3 import PPO, A2C, SAC, TD3

if __name__ == "__main__":
    train_all_scenarios(ALL_SCENARIOS, RewardOption2Wrapper, [PPO, A2C, SAC], NPPAutomationWrapper, "training_18_03")
