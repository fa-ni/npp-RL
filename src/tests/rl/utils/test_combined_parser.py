from unittest import mock

from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.obs_wrapper3 import ObservationOption3Wrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper
from stable_baselines3 import PPO


@mock.patch("src.main.rl.utils.combined_parser.WrapperMaker")
@mock.patch("src.main.rl.utils.combined_parser.parse_wrapper")
@mock.patch("src.main.rl.utils.combined_parser.parse_alg_name")
@mock.patch("src.main.rl.utils.combined_parser.parse_scenario_name")
def test_parse_information_from_path(mock_parse_scenario, mock_parse_alg, mock_parse_wrapper, mock_wrapper_maker):
    mock_parse_scenario.return_value = "scenario2"
    mock_parse_alg.return_value = "PPO"
    mock_parse_wrapper.return_value = ActionSpaceOption3Wrapper, None, ObservationOption3Wrapper, RewardOption2Wrapper
    mock_wrapper_maker.return_value = "x"
    actual = parse_information_from_path("test_path")

    assert mock_parse_scenario.call_count == 1
    assert mock_parse_alg.call_count == 1
    assert mock_parse_wrapper.call_count == 1
    assert mock_wrapper_maker.call_count == 1
    assert actual == ("src.main.rl.envs.scenario2:Scenario2", PPO, "x")
