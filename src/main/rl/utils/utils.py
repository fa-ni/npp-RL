import gym

from src.main.dto.FullReactor import FullReactor


def get_real_value(max_value: int, scaled_value: float) -> int:
    return int((scaled_value + 1) * (max_value / 2))


def is_done(full_reactor: FullReactor, length: int) -> bool:
    done = False
    if (
        full_reactor.reactor.overheated
        or full_reactor.reactor.is_blown()
        or full_reactor.generator.is_blown()
        or full_reactor.condenser.is_blown()
        or full_reactor.water_pump1.is_blown()
        or full_reactor.condenser_pump.is_blown()
        or length <= 0
    ):
        done = True
    return done


def is_done_java(backend, length: int) -> bool:
    done = False
    if (
        not backend.getReactorTankStatus()
        or not backend.getReactorStatus()
        or not backend.getTurbineStatus()
        or not backend.getCondenserStatus()
        or not backend.getWP1Status()
        or not backend.getCPStatus()
        or length <= 0
    ):
        done = True
    return done


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


def delete_env_id(env_id: str):
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if env_id in env:
            del gym.envs.registration.registry.env_specs[env]


class WrapperMaker:
    def __init__(self, action_wrapper, observation_wrapper):
        self.action_wrapper = action_wrapper
        self.observation_wrapper = observation_wrapper

    def make_wrapper(self, env):
        env = self.action_wrapper(env)
        env = self.observation_wrapper(env)
        return env
