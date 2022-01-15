import gym
from gym import register

from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env

# Configuration
from stable_baselines3.common.vec_env import VecMonitor

env_dict = gym.envs.registration.registry.env_specs.copy()
env_id = "TestEnv-v1"
for env in env_dict:
    if env_id in env:
        del gym.envs.registration.registry.env_specs[env]

register(id=env_id, entry_point="envs.env:TestEnv")  # TODO CHECK
num_cpu = 12
log_dir = "/model"


def train_agent(algorithm, environment, action_wrapper_name: str = None, obs_wrapper_name: str = None):
    try:
        environment.reset()
        model = algorithm("MlpPolicy", env=environment, verbose=1, tensorboard_log="./logs/", device="cpu",).learn(
            1500000,
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
vec_env = make_vec_env(env_id, n_envs=num_cpu)
vec_env2 = VecMonitor(vec_env)
for alg in algorithms:
    train_agent(alg, vec_env2)
