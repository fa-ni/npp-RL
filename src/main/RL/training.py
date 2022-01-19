import os

import gym
import numpy as np
from gym import register
from gym.wrappers.monitor import load_results
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import VecMonitor

from RL.wrapper.ActionSpaceOption2Wrapper import ActionSpaceOption2Wrapper
from RL.wrapper.ActionSpaceOption3Wrapper import ActionSpaceOption3Wrapper
from RL.wrapper.ObservationOption2Wrapper import ObservationOption2Wrapper
from RL.wrapper.ObservationOption3Wrapper import ObservationOption3Wrapper
from RL.wrapper.ObservationOption4Wrapper import ObservationOption4Wrapper
from RL.wrapper.ObservationOption5Wrapper import ObservationOption5Wrapper


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(
        self, check_freq, log_dir, minimum_required_reward: int = 190, number_of_last_episodes_to_check: int = 100
    ):
        super(SaveOnBestTrainingRewardCallback, self).__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.minimum_required_reward = minimum_required_reward
        self.number_of_last_episodes_to_check = number_of_last_episodes_to_check

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.number_of_last_episodes_to_check :])
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward and mean_reward > self.minimum_required_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    self.model.save(self.save_path)
        return True


# TODO Refactor
# TODO Callback to save best model
env_dict = gym.envs.registration.registry.env_specs.copy()
env_id = "TestEnv-v1"
for env in env_dict:
    if env_id in env:
        del gym.envs.registration.registry.env_specs[env]

register(id=env_id, entry_point="envs.scenario2:Scenario2")  # TODO CHECK
num_cpu = 12
log_dir = "./model"

list_of_obs_wrappers = [
    ObservationOption2Wrapper,
    ObservationOption3Wrapper,
    ObservationOption4Wrapper,
    ObservationOption5Wrapper,
]
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
        log_name = f"scenario2_{action_wrapper_name}_{obs_wrapper_name}_{algorithm.__name__}"
        callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
        environment.reset()
        model = algorithm("MlpPolicy", env=environment, verbose=1, tensorboard_log="./logs/", device="cpu",).learn(
            700000,
            tb_log_name=log_name,
            # n_eval_episodes=1,
            # eval_freq=1,
            # callback=callback
        )
    except Exception as exception:
        print("Error")
        print(exception)


# TODO add Callback to save best models during training

# Algorithms
algorithms = [A2C, PPO, DDPG]

# With Wrappers
for action_wrapper in list_of_action_wrappers:
    for observation_wrapper in list_of_obs_wrappers:
        # This is needed because make_vec_env does not allow a method with multiple parameters as wrapper_class
        wrapper_maker = WrapperMaker(action_wrapper, observation_wrapper)
        vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=wrapper_maker.make_wrapper, monitor_dir=log_dir)
        vec_env_monitor = VecMonitor(vec_env)
        for alg in algorithms:
            train_agent(alg, vec_env_monitor, action_wrapper.__name__, observation_wrapper.__name__)
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
vec_env = make_vec_env(env_id, n_envs=1)
vec_env_monitor = VecMonitor(vec_env)
for alg in algorithms:
    train_agent(alg, vec_env_monitor)
