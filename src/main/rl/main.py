from src.main.rl.training import train_all_scenarios
from src.main.rl.utils.constants import ALL_SCENARIOS

if __name__ == "__main__":
    train_all_scenarios(ALL_SCENARIOS[1:2], "reward_roof_blow_counter_pump_200_with_obs_blow_counter")
