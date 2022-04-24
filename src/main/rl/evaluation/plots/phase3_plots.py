import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd

from src.main.rl.evaluation.eval import evaluate
from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.constants import (
    scaling_factors_scenario_1,
    scaling_factors_scenario_2,
    action_dimensions_german,
    obs_scaling_factors,
    obs_dimensions,
    scaling_factors_scenario_3,
)
from src.main.rl.utils.parser import parse_scenario_name, parse_wrapper

color_mapping = {
    "standard": "#1D2D5F",
    "red": "#F65E5D",
    "orange": "#FF8800",
    "scenario1": "#1D2D5F",
    "scenario2": "#F65E5D",
    "scenario3": "#FFBC47",
    "None": "#1D2D5F",
    "ActionSpaceOption1Wrapper": "#1D2D5F",
    "NPPAutomationWrapper": "#F65E5D",
    "ActionSpaceOption2Wrapper": "#F65E5D",
    "ActionSpaceOption3Wrapper": "#FFBC47",
}


def plot_actions_taken(actions_taken: list, scenario_name: str, y_axis_scale=None) -> None:
    if y_axis_scale is None:
        y_axis_scale = [[0, 100], [0, 2200], [0, 1.2], [0, 1.2], [0, 2200]]
    action_space_size = len(actions_taken[0])
    fig, ax = plt.subplots(1, action_space_size, constrained_layout=True)
    fig.set_figheight(3)
    fig.set_figwidth(20)
    actions_positions = [[] for i in range(action_space_size)]
    # Transformation of actions taken. Final result: Each Action e.g. the first action will be togther in a list and will
    # be plotted as one graph. Total will look like the following if action_space_ length ==3:
    # [[all actions from position 0],[all actions from position 1],[all actions from position 2]]
    for item in actions_taken:
        [actions_positions[idx].append(single_action) for idx, single_action in enumerate(item)]

    # Scaling to real values as the action space is also normalized to [-1,1]
    scaled_values = []
    parsed_scenario_name = parse_scenario_name(scenario_name)
    if parsed_scenario_name == "scenario1":
        for idx, item in enumerate(actions_positions):
            # TODO double check
            scaled_values.append([int(round((x + 1) * (scaling_factors_scenario_1[idx] / 2))) for x in item])
    if parsed_scenario_name == "scenario2":
        for idx, item in enumerate(actions_positions):
            scaled_values.append(
                [-scaling_factors_scenario_2[idx] if x == 0 else scaling_factors_scenario_2[idx] for x in item]
            )
    if parsed_scenario_name == "scenario3":
        for idx, item in enumerate(actions_positions):
            scaled_values.append([scaling_factors_scenario_3[idx][x] for x in item])

    # Plotting
    [
        ax[idx].plot(action_position, color=color_mapping["standard"])
        for idx, action_position in enumerate(scaled_values)
    ]

    [
        ax[idx].xaxis.set_major_locator(plticker.MaxNLocator(5 if len(actions_positions[0]) >= 5 else 5))
        for idx in range(action_space_size)
    ]
    [ax[idx].set_title(action_dimensions_german[idx]) for idx in range(action_space_size)]
    [ax[idx].set_xlabel("Zeitschritte") for idx in range(action_space_size)]
    [ax[idx].set_ylabel("Value") for idx in range(action_space_size)]
    if y_axis_scale:
        [ax[idx].set_ylim(y_axis_scale[idx]) for idx in range(action_space_size)]

    plt.show()
    return fig
    # fig.savefig(
    #    f"src/main/rl/evaluation/plot_results/phase3_actions_plots_scen1_wo_best.png",
    #    format="png",
    #    dpi=300,
    # )


def plot_observations(observations_taken: list, y_axis_scale: list = None) -> None:
    if y_axis_scale is None:
        y_axis_scale = [
            [0, 1100],
            [0, 4200],
            [0, 570],
            [0, 6000],
            [0, 198],
            [0, 2100],
            [0, 110],
            [0, 2100],
            [0, 1.2],
            [0, 1.2],
            [0, 32],
        ]
    obs_space_size = len(observations_taken[0])
    red_critical_line_1 = {
        6: [-10, 2800, 450, 5100, 110, 4],
        11: [-10, 2800, 450, 5100, 110, -10, -10, -10, -10, -10, 4],
    }
    red_critical_line_2 = {
        6: [-10, 1200, -10, 800, -10, -10],
        11: [-10, 1200, -10, 800, -10, -10, -10, -10, -10, -10, -10],
    }
    orange_critical_line_1 = {
        6: [-10, 2500, 350, 4400, 80, 9],
        11: [-10, 2500, 350, 4400, 80, -10, -10, -10, -10, -10, 9],
    }
    orange_critical_line_2 = {
        6: [-10, 1500, -10, 1500, -10, -10],
        11: [-10, 1500, -10, 1500, -10, -10, -10, -10, -10, -10, -10],
    }
    dead_line_1 = {6: [-10, 2900, 500, 5300, 140, 0], 11: [-10, 2900, 500, 5300, 140, -10, -10, -10, -10, -10, 0]}
    dead_line_2 = {6: [-10, 1000, -10, 300, -10, -10], 11: [-10, 1000, -10, 300, -10, -10, -10, -10, -10, -10, -10]}
    fig, ax = plt.subplots(obs_space_size, 1, constrained_layout=True)
    obs_positions = [[] for i in range(obs_space_size)]
    fig.set_figheight(20)
    # fig.set_figwidth(20)

    # Transformation of actions taken. Final result: Each Action e.g. the first action will be togther in a list and will
    # be plotted as one graph. Total will look like the following if obs_space_ length ==3:
    # [[all obs from position 0],[all obs from position 1],[all obs from position 2]]
    for item in observations_taken:
        [obs_positions[idx].append(single_action) for idx, single_action in enumerate(item)]
    # Scaling
    scaled_values_obs = []
    current_obs_scaling_factors = obs_scaling_factors[obs_space_size]
    current_obs_dimensions = obs_dimensions[obs_space_size]
    for idx, item in enumerate(obs_positions):
        scaled_values_obs.append([int(round((x + 1) * (current_obs_scaling_factors[idx] / 2))) for x in item])
        # Plotting
    # print(scaled_values_obs)
    [ax[idx].plot(obs_position, color=color_mapping["standard"]) for idx, obs_position in enumerate(scaled_values_obs)]
    [ax[idx].set_ylim(y_axis_scale[idx]) for idx in range(obs_space_size)]
    [
        ax[idx].axhline(red_critical_line_1[obs_space_size][idx], color=color_mapping["red"], ls="--")
        for idx in range(obs_space_size)
    ]
    [
        ax[idx].axhline(red_critical_line_2[obs_space_size][idx], color=color_mapping["red"], ls="--")
        for idx in range(obs_space_size)
    ]
    [
        ax[idx].axhline(orange_critical_line_1[obs_space_size][idx], color=color_mapping["orange"], ls="--")
        for idx in range(obs_space_size)
    ]
    [
        ax[idx].axhline(orange_critical_line_2[obs_space_size][idx], color=color_mapping["orange"], ls="--")
        for idx in range(obs_space_size)
    ]
    [ax[idx].axhline(dead_line_1[obs_space_size][idx], color="black", ls="--") for idx in range(obs_space_size)]
    [ax[idx].axhline(dead_line_2[obs_space_size][idx], color="black", ls="--") for idx in range(obs_space_size)]
    [ax[idx].set_title(current_obs_dimensions[idx]) for idx in range(obs_space_size)]
    [ax[idx].set_xlabel("Zeitschritte") for idx in range(obs_space_size)]
    [ax[idx].yaxis.set_major_locator(plticker.MaxNLocator(5)) for idx in range(obs_space_size)]
    # [ax[idx].xaxis.set_major_locator(plticker.MaxNLocator(11)) for idx in range(obs_space_size)]
    [ax[idx].set_ylabel("Value") for idx in range(obs_space_size)]

    plt.show()
    return fig
    # fig.savefig(
    #    f"src/main/rl/evaluation/plot_results/phase3_obs_plots.png",
    #    format="png",
    #    dpi=300,
    # )


def prepare_one_combination_actions_and_obs_for_analysis(
    path_to_model: str, episode_length: int = 250, number_of_models: int = 10, obs_dimensions: int = 6
):
    # This function transforms the actions and observations for all ten models per combination so that we can easily aggregated per timestep and create
    # the corresponding statistics per timestamp and not only over the whole evaluation.
    # It returns a list of 5 dataframes. Each for one dimension of the action space. In the dataframe we have 250 rows (one per timestamp)
    # and 10 columns with the different models/runs. so if we aggregate of the columns we get the statistics for a action dimession over the 250 episodes (rows).
    # Same with different number of df for obs.
    # Also returns just a list were all actions taken and obs taken are appended.
    result_list_actions = [pd.DataFrame() for i in range(5)]
    result_list_obs = [pd.DataFrame() for i in range(obs_dimensions)]

    list_of_all_actions_taken = []
    list_of_all_obs_taken = []

    for number in range(1, number_of_models + 1):
        result_dict = {}
        full_path = path_to_model + f"_{str(number)}"
        path_to_overhand = full_path + "/best_model.zip"

        action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper = parse_wrapper(full_path)
        scenario, alg, wrapper_maker = parse_information_from_path(full_path)
        # Switch episode length here
        cum_reward, criticality_score, total_timesteps, actions_taken, obs_taken, info = evaluate(
            scenario, path_to_overhand, alg, wrapper_maker, episode_length=episode_length
        )
        # actions taken is a list of lists. The first list contains 250 further list (one per timestep). The inner list contaijs 5 values (one per dimension in the action space)
        list_of_all_actions_taken.append(actions_taken)
        list_of_all_obs_taken.append(obs_taken)
        intermediate_df_actions = pd.DataFrame(actions_taken)
        intermediate_df_obs = pd.DataFrame(obs_taken)
        result_list_actions[0] = result_list_actions[0].join(
            intermediate_df_actions[0].rename(f"action_0_run_{number}").to_frame(), how="outer"
        )
        result_list_actions[1] = result_list_actions[1].join(
            intermediate_df_actions[1].rename(f"action_1_run_{number}").to_frame(), how="outer"
        )
        result_list_actions[2] = result_list_actions[2].join(
            intermediate_df_actions[2].rename(f"action_2_run_{number}").to_frame(), how="outer"
        )
        result_list_actions[3] = result_list_actions[3].join(
            intermediate_df_actions[3].rename(f"action_3_run_{number}").to_frame(), how="outer"
        )
        result_list_actions[4] = result_list_actions[4].join(
            intermediate_df_actions[4].rename(f"action_4_run_{number}").to_frame(), how="outer"
        )

        result_list_obs[0] = result_list_obs[0].join(
            intermediate_df_obs[0].rename(f"obs_0_run_{number}").to_frame(), how="outer"
        )
        result_list_obs[1] = result_list_obs[1].join(
            intermediate_df_obs[1].rename(f"obs_1_run_{number}").to_frame(), how="outer"
        )
        result_list_obs[2] = result_list_obs[2].join(
            intermediate_df_obs[2].rename(f"obs_2_run_{number}").to_frame(), how="outer"
        )
        result_list_obs[3] = result_list_obs[3].join(
            intermediate_df_obs[3].rename(f"obs_3_run_{number}").to_frame(), how="outer"
        )
        result_list_obs[4] = result_list_obs[4].join(
            intermediate_df_obs[4].rename(f"obs_4_run_{number}").to_frame(), how="outer"
        )
        result_list_obs[5] = result_list_obs[5].join(
            intermediate_df_obs[5].rename(f"obs_5_run_{number}").to_frame(), how="outer"
        )
        if obs_dimensions >= 7:
            result_list_obs[6] = result_list_obs[6].join(
                intermediate_df_obs[6].rename(f"obs_6_run_{number}").to_frame(), how="outer"
            )
        if obs_dimensions == 11:
            result_list_obs[7] = result_list_obs[7].join(
                intermediate_df_obs[7].rename(f"obs_7_run_{number}").to_frame(), how="outer"
            )
            result_list_obs[8] = result_list_obs[8].join(
                intermediate_df_obs[8].rename(f"obs_8_run_{number}").to_frame(), how="outer"
            )
            result_list_obs[9] = result_list_obs[9].join(
                intermediate_df_obs[9].rename(f"obs_9_run_{number}").to_frame(), how="outer"
            )
            result_list_obs[10] = result_list_obs[10].join(
                intermediate_df_obs[10].rename(f"obs_10_run_{number}").to_frame(), how="outer"
            )
    return result_list_actions, result_list_obs, list_of_all_actions_taken, list_of_all_obs_taken
