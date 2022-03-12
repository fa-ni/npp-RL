from src.main.rl.utils.parser import parse_wrapper, parse_scenario_name, parse_alg_name
from src.main.rl.wrapper.action_wrapper2 import ActionSpaceOption2Wrapper
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.obs_wrapper2 import ObservationOption2Wrapper
from src.main.rl.wrapper.obs_wrapper3 import ObservationOption3Wrapper
from src.main.rl.wrapper.obs_wrapper4 import ObservationOption4Wrapper
from src.main.rl.wrapper.obs_wrapper5 import ObservationOption5Wrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper
from src.main.rl.wrapper.reward_wrapper3 import RewardOption3Wrapper
import pytest


@pytest.mark.parametrize(
    ["input", "action", "obs", "reward"],
    [
        [
            "scenario1ActionSpaceOption2WrapperObservationOption2WrapperRewardOption2Wrapper",
            ActionSpaceOption2Wrapper,
            ObservationOption2Wrapper,
            RewardOption2Wrapper,
        ],
        [
            "scenario1ActionSpaceOption3WrapperObservationOption3WrapperRewardOption3Wrapper",
            ActionSpaceOption3Wrapper,
            ObservationOption3Wrapper,
            RewardOption3Wrapper,
        ],
        ["scenario1NoneWrapperObservationOption4WrapperRewarder", None, ObservationOption4Wrapper, None],
        [
            "scenario1ActionSpaceOption2WrapperObservationOption5WrapperRewardOption2Wrapper",
            ActionSpaceOption2Wrapper,
            ObservationOption5Wrapper,
            RewardOption2Wrapper,
        ],
        ["", None, None, None],
        [
            "scenario1ActionSpaceOption2WrapperObservationOption2WrapperRewardOption2WrapperActionSpaceOption3",
            ActionSpaceOption2Wrapper,
            ObservationOption2Wrapper,
            RewardOption2Wrapper,
        ],
    ],
)
def test_parse_wrapper(input, action, obs, reward):
    actual = parse_wrapper(input)
    assert actual == (action, obs, reward)


@pytest.mark.parametrize(
    ["input", "expected"],
    [
        ["scenario1xx", "scenario1"],
        ["xxscenario2", "scenario2"],
        ["0scenario3", "scenario3"],
        ["scenario10senario2", "scenario1"],
    ],
)
def test_parse_scenario_name(input, expected):
    actual = parse_scenario_name(input)
    assert actual == expected


def test_test_parse_scenario_name_raises():
    with pytest.raises(Exception):
        parse_scenario_name("Raise?")


@pytest.mark.parametrize(
    ["input", "expected"],
    [["PPOx", "PPO"], ["DDPG0", "DDPG"], ["PPODDPG", "DDPG"], ["TD3ido", "TD3"], ["SACSAC", "SAC"], ["xA2C3", "A2C"]],
)
def test_parse_alg_name(input, expected):
    actual = parse_alg_name(input)
    assert actual == expected


def test_parse_alg_name_raises():
    with pytest.raises(Exception):
        parse_alg_name("Raise?")
