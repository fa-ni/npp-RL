import gym
import matplotlib.pyplot as plt
from gym import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.main.rl.utils.utils import WrapperMaker
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.obs_wrapper4 import ObservationOption4Wrapper
from src.main.rl.wrapper.obs_wrapper5 import ObservationOption5Wrapper

scenarios = ["envs.scenario2:Scenario2"]


def plot_actions_taken(actions_taken: list) -> None:
    fig, ax = plt.subplots(1, 5)
    # TODO REFACTOR
    action1 = [item[0][0] for item in actions_taken]
    action2 = [item[0][1] for item in actions_taken]
    action3 = [item[0][2] for item in actions_taken]
    action4 = [item[0][3] for item in actions_taken]
    action5 = [item[0][4] for item in actions_taken]

    ax[0].plot(action1)
    ax[1].plot(action2)
    ax[2].plot(action3)
    ax[3].plot(action4)
    ax[4].plot(action5)
    plt.show()


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
    model = PPO.load(
        "models/scenario2/reward_roof_blow_counter_pump_30_with_obs_blow_counter/scenario2_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_PPO_reward_roof_blow_counter_pump_30_with_obs_blow_counter/best_model.zip"
    )
    obs = vec_env.reset()
    actions_taken = []

    for _ in range(1000):
        vec_env.render()

        # Evaluate
        # Interesting if deterministic==True, we always get the same reward, otherwise not
        action, _states = model.predict(obs, deterministic=True)
        actions_taken.append(action)
        obs, reward, done, info = vec_env.step(action)
        mean_reward_over_multiple_evaluations.append(reward)
        if done:

            # plot_actions_taken(actions_taken)
            print(sum(mean_reward_over_multiple_evaluations))
            mean_reward_over_multiple_evaluations = []
            done = False
            obs = vec_env.reset()
            actions_taken = []
