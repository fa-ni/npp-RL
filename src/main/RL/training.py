import gym
from gym import register

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

# Configuration
env_dict = gym.envs.registration.registry.env_specs.copy()
env_id = "TestEnv-v1"
for env in env_dict:
    if env_id in env:
        del gym.envs.registration.registry.env_specs[env]

register(id=env_id, entry_point="envs.env:TestEnv")  # TODO CHECK
num_cpu = 12
vec_env = make_vec_env(env_id, n_envs=num_cpu)
log_dir = "/model"

# Algorithms
algorithms = [PPO, A2C, DDPG]

# Training
for item in algorithms:
    model = item("MlpPolicy", env=vec_env, verbose=1, tensorboard_log="./logs/", device="cpu",).learn(
        500000,
        tb_log_name=f"further_multibinary_{item.__name__}",
        log_interval=100,
        # n_eval_episodes=1,
        # eval_freq=1,
    )
