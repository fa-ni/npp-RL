import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from src.main.rl.utils.constants import (
    scaling_factors_scenario_1,
    scaling_factors_scenario2,
    action_dimensions_german,
    obs_scaling_factors,
    obs_dimensions,
    scaling_factors_scenario3,
)
from src.main.rl.utils.parser import parse_scenario_name

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
                [-scaling_factors_scenario2[idx] if x == 0 else scaling_factors_scenario2[idx] for x in item]
            )
    if parsed_scenario_name == "scenario3":
        for idx, item in enumerate(actions_positions):
            scaled_values.append([scaling_factors_scenario3[idx][x] for x in item])

    # Plotting
    [
        ax[idx].plot(action_position, color=color_mapping["standard"])
        for idx, action_position in enumerate(scaled_values)
    ]
    [ax[idx].xaxis.set_major_locator(plticker.MaxNLocator(11)) for idx in range(action_space_size)]
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
            [0, 1000],
            [0, 4000],
            [0, 550],
            [0, 8000],
            [0, 180],
            [0, 2000],
            [0, 100],
            [0, 2000],
            [0, 1.2],
            [0, 1.2],
            [0, 36],
        ]
    obs_space_size = len(observations_taken[0])
    red_critical_line_1 = [-10, 2800, 450, 5100, 110, -10, -10, -10, -10, -10, 4]
    red_critical_line_2 = [-10, 1200, -10, 800, -10, -10, -10, -10, -10, -10, -10]
    orange_critical_line_1 = [-10, 2500, 350, 4400, 80, -10, -10, -10, -10, -10, 9]
    orange_critical_line_2 = [-10, 1500, -10, 1500, -10, -10, -10, -10, -10, -10, -10]
    dead_line_1 = [-10, 2900, 500, 5300, 140, 0]
    dead_line_2 = [-10, 1000, -10, 300, -10, -10]
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
    [ax[idx].axhline(red_critical_line_1[idx], color=color_mapping["red"], ls="--") for idx in range(obs_space_size)]
    [ax[idx].axhline(red_critical_line_2[idx], color=color_mapping["red"], ls="--") for idx in range(obs_space_size)]
    [
        ax[idx].axhline(orange_critical_line_1[idx], color=color_mapping["orange"], ls="--")
        for idx in range(obs_space_size)
    ]
    [
        ax[idx].axhline(orange_critical_line_2[idx], color=color_mapping["orange"], ls="--")
        for idx in range(obs_space_size)
    ]
    [ax[idx].axhline(dead_line_1[idx], color="black", ls="--") for idx in range(obs_space_size)]
    [ax[idx].axhline(dead_line_2[idx], color="black", ls="--") for idx in range(obs_space_size)]
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
