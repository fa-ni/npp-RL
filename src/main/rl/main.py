from src.main.rl.training import train_all_scenarios
from src.main.rl.utils.constants import ALL_SCENARIOS
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper

if __name__ == "__main__":
    for _ in range(2):
        train_all_scenarios(ALL_SCENARIOS, RewardOption2Wrapper, "full_test_26_02_wo_automation")
