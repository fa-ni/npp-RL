import gym
import mock
from stable_baselines3 import PPO, A2C, SAC, TD3

from src.main.rl.utils.constants import ALL_SCENARIOS
from src.main.rl.training import train_all_scenarios
from src.main.rl.wrapper.npp_automation_wrapper import NPPAutomationWrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper


@mock.patch("src.main.rl.training.train_agent")
def test_number_of_models_with_one_envs(mock_train_agent):
    scenarios = ["src.main.rl.envs.scenario2:Scenario2"]
    train_all_scenarios(ALL_SCENARIOS, RewardOption2Wrapper, [PPO, A2C], NPPAutomationWrapper, "training_18_03")
    # Count is the following:
    # 3 (ActionWrapper) * 5 (ObservationWrappers) * 2 (Alg) * 1 (NPPAutomation) = 45
    assert mock_train_agent.call_count == 90
    # Resetting all registered envs in gym
    gym.envs.registration.registry.env_specs.clear()


@mock.patch("src.main.rl.training.train_agent")
def test_number_of_models_with_three_envs(mock_train_agent):
    train_all_scenarios(
        ALL_SCENARIOS, RewardOption2Wrapper, [PPO, A2C, SAC, TD3], NPPAutomationWrapper, "training_18_03"
    )
    # 3 (ActionSpace) * 5 (ObsSpace) * 3 (Scenarios) = 45
    # 45 * 4 (Algos) = 180 * 1 (NPPAutomation) = 180
    # Note the method is called 360 times as there are 360 possible
    # combinations. In real not TD3 and SAC can only be used with
    # scenario1 which leads to fewer runs later.
    assert mock_train_agent.call_count == 180
