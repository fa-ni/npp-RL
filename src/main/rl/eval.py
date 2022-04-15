import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from gym import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.main.rl.analysis.criticality_helper import calculate_criticality_score_with_reward_functions
from src.main.rl.envs.sop import get_actions_sop
from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.constants import (
    scaling_factors_scenario_1,
    scaling_factors_scenario2,
    action_dimensions,
    obs_scaling_factors,
    obs_dimensions,
)
from src.main.rl.utils.parser import parse_scenario_name
from src.main.rl.utils.utils import WrapperMaker, delete_env_id
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper


def plot_actions_taken(actions_taken: list, scenario_name: str) -> None:
    action_space_size = len(actions_taken[0][0])
    fig, ax = plt.subplots(action_space_size, 1, constrained_layout=True)
    fig.set_figheight(10)
    actions_positions = [[] for i in range(action_space_size)]
    # Transformation of actions taken. Final result: Each Action e.g. the first action will be togther in a list and will
    # be plotted as one graph. Total will look like the following if action_space_ length ==3:
    # [[all actions from position 0],[all actions from position 1],[all actions from position 2]]
    for item in actions_taken:
        [actions_positions[idx].append(single_action) for idx, single_action in enumerate(item[0])]

    # Scaling to real values as the action space is also normalized to [-1,1]
    scaled_values = []
    parsed_scenario_name = parse_scenario_name(scenario_name)
    if parsed_scenario_name == "scenario1":
        for idx, item in enumerate(actions_positions):
            scaled_values.append([int(round((x + 1) * (scaling_factors_scenario_1[idx] / 2))) for x in item])
    if parsed_scenario_name == "scenario2":
        for idx, item in enumerate(actions_positions):
            scaled_values.append(
                [-scaling_factors_scenario2[idx] if x == 0 else scaling_factors_scenario2[idx] for x in item]
            )
    # TODO if needed at scaled values for scenario 3

    # Plotting
    [ax[idx].plot(action_position) for idx, action_position in enumerate(scaled_values)]
    [ax[idx].xaxis.set_major_locator(plticker.MaxNLocator(11)) for idx in range(action_space_size)]
    [ax[idx].set_title(action_dimensions[idx]) for idx in range(action_space_size)]
    [ax[idx].set_xlabel("Episode") for idx in range(action_space_size)]
    [ax[idx].set_ylabel("Value") for idx in range(action_space_size)]
    plt.show()


def plot_observations(observations_taken: list) -> None:
    obs_space_size = len(observations_taken[0][0])
    fig, ax = plt.subplots(obs_space_size, 1, constrained_layout=True)
    obs_positions = [[] for i in range(obs_space_size)]
    fig.set_figheight(20)
    # Transformation of actions taken. Final result: Each Action e.g. the first action will be togther in a list and will
    # be plotted as one graph. Total will look like the following if obs_space_ length ==3:
    # [[all obs from position 0],[all obs from position 1],[all obs from position 2]]
    for item in observations_taken:
        [obs_positions[idx].append(single_action) for idx, single_action in enumerate(item[0])]
    # Scaling
    scaled_values_obs = []
    current_obs_scaling_factors = obs_scaling_factors[obs_space_size]
    current_obs_dimensions = obs_dimensions[obs_space_size]
    for idx, item in enumerate(obs_positions):
        scaled_values_obs.append([int(round((x + 1) * (current_obs_scaling_factors[idx] / 2))) for x in item])
        # Plotting
    [ax[idx].plot(obs_position) for idx, obs_position in enumerate(scaled_values_obs)]
    [ax[idx].set_title(current_obs_dimensions[idx]) for idx in range(obs_space_size)]
    [ax[idx].set_xlabel("Episode") for idx in range(obs_space_size)]
    [ax[idx].yaxis.set_major_locator(plticker.MaxNLocator(5)) for idx in range(obs_space_size)]
    [ax[idx].xaxis.set_major_locator(plticker.MaxNLocator(11)) for idx in range(obs_space_size)]
    [ax[idx].set_ylabel("Value") for idx in range(obs_space_size)]
    plt.show()


def evaluate(
    scenario_name: str,
    path: str,
    alg: OnPolicyAlgorithm,
    wrapper: WrapperMaker,
    starting_state=None,
    episode_lgenth: int = 250,
) -> float:
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(
        env_id,
        n_envs=1,
        wrapper_class=wrapper.make_wrapper,
        env_kwargs={"starting_state": starting_state, "length": episode_lgenth},
    )

    mean_reward_over_multiple_evaluations = []
    model = alg.load(path)
    obs = vec_env.reset()
    actions_taken = []
    observations_taken = []
    reactor_status_over_time = []
    observations_taken.append(obs)
    for _ in range(250):
        # vec_env.render()

        action, _states = model.predict(obs, deterministic=True)
        actions_taken.append(action)
        obs, reward, done, info = vec_env.step(action)
        reactor_status_over_time.append(info[0])
        observations_taken.append(obs)
        mean_reward_over_multiple_evaluations.append(reward)
        if done:
            plot_actions_taken(actions_taken, scenario_name)
            plot_observations(observations_taken)

            print(f"Criticality Score1: {calculate_criticality_score_with_reward_functions(reactor_status_over_time)}")
            # criticality_of_states = prepare_critical_states_analysis(reactor_status_over_time)
            # print(f"Criticality Score2: {calculate_score(criticality_of_states)}")
            # print(sum(mean_reward_over_multiple_evaluations))
            result = sum(mean_reward_over_multiple_evaluations)
            print(result)
            mean_reward_over_multiple_evaluations = []
            obs = vec_env.reset()
            actions_taken = []
            observations_taken = []
            return result


def evaluate_sop():
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point="envs.scenario1:Scenario1")
    wrapper_maker = WrapperMaker(ActionSpaceOption3Wrapper, None, None, RewardOption2Wrapper)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=wrapper_maker.make_wrapper)
    mean_reward_over_multiple_evaluations = []
    obs = vec_env.reset()
    actions_taken = []
    observations_taken = []
    observations_taken.append(obs)
    for i in range(250):
        # vec_env.render()
        # Evaluate
        # Interesting if deterministic==True, we always get the same reward, otherwise not
        action = np.array([get_actions_sop(249 - i)])
        actions_taken.append(action)
        obs, reward, done, info = vec_env.step(action)
        observations_taken.append(obs)
        mean_reward_over_multiple_evaluations.append(reward)
        if done:
            # plot_actions_taken(actions_taken, scenario_name)
            plot_observations(observations_taken)

            print(sum(mean_reward_over_multiple_evaluations))
            result = sum(mean_reward_over_multiple_evaluations)
            mean_reward_over_multiple_evaluations = []
            obs = vec_env.reset()
            actions_taken = []
            observations_taken = []
            return result


# evaluate_sop()
path = "scenario1/training_18_03/scenario1_ActionSpaceOption3Wrapper_ObservationOption4Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_TD3_training_18_03_2/best_model.zip"
scenario, alg, wrapper_maker = parse_information_from_path(path)
evaluate(
    scenario,
    path,
    alg,
    wrapper_maker,
)  # starting_state=create_starting_state_option3a(), episode_lgenth=250)
