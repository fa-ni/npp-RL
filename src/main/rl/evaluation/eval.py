import numpy as np
from gym import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.main.rl.evaluation.criticality_helper import calculate_criticality_score_with_reward_functions
from src.main.rl.evaluation.sop import get_actions_sop
from src.main.rl.utils.utils import WrapperMaker, delete_env_id
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.obs_wrapper4 import ObservationOption4Wrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper


def evaluate(
    scenario_name: str,
    path: str,
    alg: OnPolicyAlgorithm,
    wrapper: WrapperMaker,
    starting_state=None,
    episode_length: int = 250,
) -> [float, list, int, list, list, dict]:
    """
    The evaluate function is used to evaluate a trained agent.
    It returns a lot of information about like return, criticality score, observations, actions taken and
    more information.
    """
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(
        env_id,
        n_envs=1,
        wrapper_class=wrapper.make_wrapper,
        env_kwargs={"starting_state": starting_state, "length": episode_length},
    )

    rewards_per_timestep = []
    model = alg.load(path)
    obs = vec_env.reset()
    actions_taken = []
    observations_taken = []
    reactor_status_over_time = []
    for i in range(episode_length):
        action, _states = model.predict(obs, deterministic=True)
        # Add action for timestep 0 -> Action is always the most neutral one
        if i == 0:
            observations_taken.append(obs[0])
            if scenario_name == "src.main.rl.envs.scenario1:Scenario1":
                actions_taken.append([-1 for i in range(len(action[0]))])
            elif scenario_name == "src.main.rl.envs.scenario2:Scenario2":
                actions_taken.append([0 for i in range(len(action[0]))])
            elif scenario_name == "src.main.rl.envs.scenario3:Scenario3":
                actions_taken.append([4 for i in range(len(action[0]))])
        actions_taken.append(action[0])
        obs, reward, done, info = vec_env.step(action)
        if done:
            observations_taken.append(info[0]["terminal_observation"])
        else:
            observations_taken.append(obs[0])
        reactor_status_over_time.append(info[0])
        rewards_per_timestep.append(reward)
        if done:
            single_min_crit_score, combined_score_min = calculate_criticality_score_with_reward_functions(
                reactor_status_over_time
            )
            result = sum(rewards_per_timestep)
            print(f"Return: {result[0]}")
            obs = vec_env.reset()
            return (
                result[0],
                [single_min_crit_score, combined_score_min],
                i + 1,
                actions_taken,
                observations_taken,
                info,
            )


def evaluate_terminal_state_obs(
    scenario_name: str,
    path: str,
    alg: OnPolicyAlgorithm,
    wrapper: WrapperMaker,
    starting_state=None,
    episode_length: int = 250,
) -> [float, float]:
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(
        env_id,
        n_envs=1,
        wrapper_class=wrapper.make_wrapper,
        env_kwargs={"starting_state": starting_state, "length": episode_length},
    )

    rewards_per_timestep = []
    model = alg.load(path)
    obs = vec_env.reset()
    observations_taken = []
    for i in range(episode_length):
        # vec_env.render()
        action, _states = model.predict(obs, deterministic=True)
        if i == 0:
            observations_taken.append(obs[0])
        obs, reward, done, info = vec_env.step(action)
        if done:
            observations_taken.append(info[0]["terminal_observation"])
        else:
            normalized_power_output = 2 * (info[0]["Power_Output"] / 800) - 1
            normalized_reactor_water_level = 2 * (info[0]["Reactor_WaterLevel"] / 4000) - 1
            normalized_reactor_pressure = 2 * (info[0]["Reactor_Pressure"] / 550) - 1
            normalized_condenser_water_level = 2 * (info[0]["Condenser_WaterLevel"] / 8000) - 1
            normalized_condenser_pressure = 2 * (info[0]["Condenser_Pressure"] / 180) - 1
            normalized_blow_counter = 2 * (info[0]["Blow_Counter"] / 30) - 1

            obs_from_info = [
                normalized_power_output,
                normalized_reactor_water_level,
                normalized_reactor_pressure,
                normalized_condenser_water_level,
                normalized_condenser_pressure,
                normalized_blow_counter,
            ]
            observations_taken.append(obs_from_info)

        rewards_per_timestep.append(reward)
        if done:
            # print(observations_taken)
            # observations_taken.append(obs[0])
            # plot_actions_taken(actions_taken, scenario_name)
            # plot_observations(observations_taken[:-1])
            # criticality_of_states = prepare_critical_states_analysis(reactor_status_over_time)
            # print(f"Criticality Score2: {calculate_score(criticality_of_states)}")
            # print(sum(mean_reward_over_multiple_evaluations))
            result = sum(rewards_per_timestep)
            print(f"Return: {result[0]}")
            obs = vec_env.reset()
            return result[0], observations_taken, info


def evaluate_sop() -> [float, list, int, list, list, dict]:
    """
    This function is used to evaluate the SOP policy. It returns the same information as the normal
    evaluate function for the agents.
    """
    env_id = "TestEnv-v1"
    delete_env_id(env_id)
    scenario_name = "scenario1"
    register(id=env_id, entry_point="src.main.rl.envs.scenario1:Scenario1")
    wrapper_maker = WrapperMaker(ActionSpaceOption3Wrapper, None, ObservationOption4Wrapper, RewardOption2Wrapper)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=wrapper_maker.make_wrapper)
    rewards_per_timestep = []
    obs = vec_env.reset()
    actions_taken = []
    observations_taken = []
    reactor_status_over_time = []

    for i in range(250):
        observations_taken.append(obs[0])
        action = np.array([get_actions_sop(250 - i)])
        obs, reward, done, info = vec_env.step(action)
        actions_taken.append(action[0])
        if i == 5:
            print("x")
        reactor_status_over_time.append(info[0])
        rewards_per_timestep.append(reward)
        if done:
            # plot_actions_taken(actions_taken, scenario_name)
            # plot_observations(observations_taken[:-1])
            single_min_crit_score, combined_score_min = calculate_criticality_score_with_reward_functions(
                reactor_status_over_time
            )
            result = sum(rewards_per_timestep)
            print(f"Return: {result[0]}")
            obs = vec_env.reset()
            return result, [single_min_crit_score, combined_score_min], i + 1, actions_taken, observations_taken, info


def get_single_reward_sop() -> list:
    env_id = "TestEnv-v1"
    delete_env_id(env_id)
    register(id=env_id, entry_point="src.main.rl.envs.scenario1:Scenario1")
    wrapper_maker = WrapperMaker(ActionSpaceOption3Wrapper, None, ObservationOption4Wrapper, RewardOption2Wrapper)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=wrapper_maker.make_wrapper)

    rewards_per_timestep = []
    obs = vec_env.reset()

    for i in range(250):
        action = np.array([get_actions_sop(250 - i)])
        obs, reward, done, info = vec_env.step(action)
        rewards_per_timestep.append(reward[0])
        if done:
            return rewards_per_timestep[:-1]


def get_single_reward(
    scenario_name: str,
    path: str,
    alg: OnPolicyAlgorithm,
    wrapper: WrapperMaker,
    starting_state=None,
    episode_length: int = 250,
) -> list:
    env_id = "TestEnv-v1"
    delete_env_id(env_id)
    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(
        env_id,
        n_envs=1,
        wrapper_class=wrapper.make_wrapper,
        env_kwargs={"starting_state": starting_state, "length": episode_length},
    )

    rewards_per_timestep = []
    model = alg.load(path)
    obs = vec_env.reset()

    for i in range(episode_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        rewards_per_timestep.append(reward[0])
        if done:
            return rewards_per_timestep[:-1]


# evaluate_sop()
# path = "../models/scenario1/training_04_06/scenario1_ActionSpaceOption3Wrapper_ObservationOption4Wrapper_None_RewardOption2Wrapper_TD3_training_04_06_5/best_model.zip"
###path = "../models/scenario3/training_04_06/scenario3_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_PPO_training_04_06_1/best_model.zip"

####
#### path = "../models/models/scenario2/training_04_06/scenario2_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_None_RewardOption2Wrapper_PPO_training_04_06_1/best_model.zip"
####
# scenario, alg, wrapper_maker = parse_information_from_path(path)
# evaluate(
#   scenario,
#   path,
#   alg,
#   wrapper_maker,episode_length=250
# )  # starting_state=create_starting_state_option3a(), episode_lgenth=250)
