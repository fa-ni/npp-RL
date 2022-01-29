import gym
import mock

from src.main.RL.utils.constants import ALL_SCENARIOS
from src.main.RL.training import train_all_scenarios


@mock.patch("src.main.RL.training.train_agent")
def test_number_of_models_with_one_envs(mock_train_agent):
    scenarios = ["src.main.RL.envs.scenario2:Scenario2"]
    len_standard_env_dict = len(gym.envs.registration.registry.env_specs)
    train_all_scenarios(scenarios)
    # Count is the following:
    # 2 (ActionWrapper) * 4 (ObservationWrappers) * 3 (Alg) = 24
    # 2 (ActionWrapper) without Obs * 3 (Alg) = 6
    # 4 (ObservationWrappers) without Action * 3 (Alg) = 12
    # 1 without Action and without Obs * 3 (Alg) = 3
    # 24+6+12+3=45
    assert mock_train_agent.call_count == 45
    # And 45 / 3 (Alg) = 15 -> 15 different scenarios with/without Wrappers
    assert len_standard_env_dict + 15 == len(gym.envs.registration.registry.env_specs)
    # Resetting all registered envs in gym
    gym.envs.registration.registry.env_specs.clear()


@mock.patch("src.main.RL.training.train_agent")
def test_number_of_models_with_three_envs(mock_train_agent):
    len_standard_env_dict = len(gym.envs.registration.registry.env_specs)
    train_all_scenarios(ALL_SCENARIOS)
    assert mock_train_agent.call_count == 45 * 3
    # And 135 / 3 (Alg) = 45 -> 45 different scenarios with/without Wrappers
    assert len_standard_env_dict + 45 == len(gym.envs.registration.registry.env_specs)
