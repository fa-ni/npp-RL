import gym
from gym import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from RL.utils.utils import WrapperMaker
from RL.wrapper.ActionSpaceOption3Wrapper import ActionSpaceOption3Wrapper
from RL.wrapper.ObservationOption5Wrapper import ObservationOption5Wrapper

scenarios = ["envs.scenario2:Scenario2"]
for scenario in scenarios:
    env_dict = gym.envs.registration.registry.env_specs.copy()
    env_id = "TestEnv-v1"
    for env in env_dict:
        if env_id in env:
            del gym.envs.registration.registry.env_specs[env]

    register(id=env_id, entry_point=scenario)
    x = WrapperMaker(ActionSpaceOption3Wrapper, ObservationOption5Wrapper)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=x.make_wrapper)

    mean_reward_over_multiple_evaluations = []
    model = PPO.load("./best_model/best_model.zip")
    obs = vec_env.reset()
    for _ in range(10000):
        # Evaluate
        # Interesting if deterministic==True, we always get the same reward, otherwise not
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        mean_reward_over_multiple_evaluations.append(reward)
        if done:
            print(sum(mean_reward_over_multiple_evaluations))
            mean_reward_over_multiple_evaluations = []
            done = False
            obs = vec_env.reset()
