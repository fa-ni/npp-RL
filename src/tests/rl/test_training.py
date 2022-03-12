import gym
import mock

from src.main.rl.utils.constants import ALL_SCENARIOS
from src.main.rl.training import train_all_scenarios
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper


@mock.patch("src.main.rl.training.train_agent")
def test_number_of_models_with_one_envs(mock_train_agent):
    scenarios = ["src.main.rl.envs.scenario2:Scenario2"]
    train_all_scenarios(scenarios, RewardOption2Wrapper)
    # Count is the following:
    # 3 (ActionWrapper) * 5 (ObservationWrappers) * 4 (Alg) * 2 (NPPAutomation) = 120
    assert mock_train_agent.call_count == 120
    # Resetting all registered envs in gym
    gym.envs.registration.registry.env_specs.clear()


@mock.patch("src.main.rl.training.train_agent")
def test_number_of_models_with_three_envs(mock_train_agent):
    train_all_scenarios(ALL_SCENARIOS, RewardOption2Wrapper)
    # 3 (ActionSpace) * 5 (ObsSpace) * 3 (Scenarios) = 45
    # 45 * 4 (Algos) = 180 * 2 (NPPAutomation) = 360
    # Note the method is called 360 times as there are 360 possible
    # combinations. In real not TD3 and SAC can only be used with
    # scenario1 which leads to fewer runs later.
    assert mock_train_agent.call_count == 360
