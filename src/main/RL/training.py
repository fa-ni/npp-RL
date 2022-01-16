import gym
from gym import register
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from RL.wrapper.ActionSpaceOption2Wrapper import ActionSpaceOption2Wrapper
from RL.wrapper.ActionSpaceOption3Wrapper import ActionSpaceOption3Wrapper
from RL.wrapper.ObservationOption2Wrapper import ObservationOption2Wrapper
from RL.wrapper.ObservationOption3Wrapper import ObservationOption3Wrapper
from RL.wrapper.ObservationOption4Wrapper import ObservationOption4Wrapper
from RL.wrapper.ObservationOption5Wrapper import ObservationOption5Wrapper

# TODO Refactor
# TODO Callback to save best model
env_dict = gym.envs.registration.registry.env_specs.copy()
env_id = "TestEnv-v1"
for env in env_dict:
    if env_id in env:
        del gym.envs.registration.registry.env_specs[env]

register(id=env_id, entry_point="envs.scenario2:Scenario2")  # TODO CHECK
num_cpu = 12
log_dir = "/model"

list_of_obs_wrappers = [ObservationOption2Wrapper, ObservationOption4Wrapper, ObservationOption5Wrapper]  # TODO
list_of_action_wrappers = [ActionSpaceOption2Wrapper, ActionSpaceOption3Wrapper]


class WrapperMaker:
    def __init__(self, action_wrapper, observation_wrapper):
        self.action_wrapper = action_wrapper
        self.observation_wrapper = observation_wrapper

    def make_wrapper(self, env):
        env = self.action_wrapper(env)
        env = self.observation_wrapper(env)
        return env


def train_agent(algorithm, environment, action_wrapper_name: str = None, obs_wrapper_name: str = None):
    try:
        environment.reset()
        model = algorithm("MlpPolicy", env=environment, verbose=1, tensorboard_log="./logs/", device="cpu",).learn(
            700000,
            tb_log_name=f"scenario2_{action_wrapper_name}_{obs_wrapper_name}_{algorithm.__name__}",
            # n_eval_episodes=1,
            # eval_freq=1,
            # callback=event_callback
        )
    except Exception as exception:
        print("Error")
        print(exception)


# Algorithms
algorithms = [A2C, PPO, DDPG]

# With Wrappers
for action_wrapper in list_of_action_wrappers:
    for observation_wrapper in list_of_obs_wrappers:
        # This is needed because make_vec_env does not allow a method with multiple parameters as wrapper_class
        wrapper_maker = WrapperMaker(action_wrapper, observation_wrapper)
        vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=wrapper_maker.make_wrapper)
        vec_env_monitor = VecMonitor(vec_env)
        for alg in algorithms:
            train_agent(alg, vec_env_monitor, action_wrapper.__name__, observation_wrapper.__name__)
## TODO TEST BEFORE RUNNING ##
# Single Wrapper
for action_wrapper in list_of_action_wrappers:
    vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=action_wrapper)
    vec_env_monitor = VecMonitor(vec_env)
    for alg in algorithms:
        train_agent(alg, vec_env_monitor, action_wrapper.__name__, None)
for observation_wrapper in list_of_obs_wrappers:
    vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=observation_wrapper)
    vec_env_monitor = VecMonitor(vec_env)
    for alg in algorithms:
        train_agent(alg, vec_env_monitor, None, observation_wrapper.__name__)

# Without Wrappers
wrapper_maker = WrapperMaker(ActionSpaceOption2Wrapper, ObservationOption3Wrapper)
vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=wrapper_maker.make_wrapper)
vec_env_monitor = VecMonitor(vec_env)
for alg in algorithms:
    train_agent(alg, vec_env_monitor)
