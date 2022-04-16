from src.main.rl.utils.constants import alg_mapping
from src.main.rl.utils.parser import parse_scenario_name, parse_alg_name, parse_wrapper
from src.main.rl.utils.utils import WrapperMaker


def parse_information_from_path(path: str):
    scenario_name = parse_scenario_name(path)
    result_scenario = f"src.main.rl.envs.{scenario_name}:{scenario_name.capitalize()}"
    alg = alg_mapping[parse_alg_name(path)]
    action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper = parse_wrapper(path)
    wrapper_maker = WrapperMaker(action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper)
    return result_scenario, alg, wrapper_maker
