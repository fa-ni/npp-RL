from src.main.rl.wrapper.action_wrapper2 import ActionSpaceOption2Wrapper
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.obs_wrapper2 import ObservationOption2Wrapper
from src.main.rl.wrapper.obs_wrapper3 import ObservationOption3Wrapper
from src.main.rl.wrapper.obs_wrapper4 import ObservationOption4Wrapper
from src.main.rl.wrapper.obs_wrapper5 import ObservationOption5Wrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper
from src.main.rl.wrapper.reward_wrapper3 import RewardOption3Wrapper


def parse_wrapper(string: str):
    if "ActionSpaceOption2Wrapper" in string:
        action_result = ActionSpaceOption2Wrapper
    elif "ActionSpaceOption3Wrapper" in string:
        action_result = ActionSpaceOption3Wrapper
    else:
        action_result = None
    if "ObservationOption2Wrapper" in string:
        obs_result = ObservationOption2Wrapper
    elif "ObservationOption3Wrapper" in string:
        obs_result = ObservationOption3Wrapper
    elif "ObservationOption4Wrapper" in string:
        obs_result = ObservationOption4Wrapper
    elif "ObservationOption5Wrapper" in string:
        obs_result = ObservationOption5Wrapper
    else:
        obs_result = None
    if "RewardOption2Wrapper" in string:
        reward_result = RewardOption2Wrapper
    elif "RewardOption3Wrapper" in string:
        reward_result = RewardOption3Wrapper
    else:
        reward_result = None
    return action_result, obs_result, reward_result


def parse_scenario_name(scenario: str) -> str:
    if "scenario1" in scenario:
        result = "scenario1"
    elif "scenario2" in scenario:
        result = "scenario2"
    elif "scenario3" in scenario:
        result = "scenario3"
    else:
        raise Exception("Not able to parse scenario name.")
    return result


def parse_alg_name(string: str) -> str:
    if "A2C" in string:
        result = "A2C"
    elif "DDPG" in string:
        result = "DDPG"
    elif "PPO" in string:
        result = "PPO"
    elif "TD3" in string:
        result = "TD3"
    elif "SAC" in string:
        result = "SAC"
    else:
        raise Exception("Algorithm name not in string.")
    return result
