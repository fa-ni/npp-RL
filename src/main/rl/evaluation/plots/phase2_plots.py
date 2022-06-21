import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import re

from src.main.rl.utils.parser import parse_alg_name
from src.main.rl.utils.utils import parse_category

colors_to_use = ["#1D2D5F", "#F65E5D", "#FFBC47", "#40CEE3"]

color_mapping = {
    "scenario1": "#1D2D5F",
    "scenario2": "#F65E5D",
    "scenario3": "#FFBC47",
    "None": "#1D2D5F",
    "ActionSpaceOption1Wrapper": "#1D2D5F",
    "NPPAutomationWrapper": "#F65E5D",
    "ActionSpaceOption2Wrapper": "#F65E5D",
    "ActionSpaceOption3Wrapper": "#FFBC47",
    "<class 'stable_baselines3.sac.sac.SAC'>": colors_to_use[0],
    "<class 'stable_baselines3.td3.td3.TD3'>": colors_to_use[1],
    "<class 'stable_baselines3.a2c.a2c.A2C'>": colors_to_use[2],
    "<class 'stable_baselines3.ppo.ppo.PPO'>": colors_to_use[3],
}

# Used for showing that random seeds have a significant influence on the result
def create_multi_object_plot(df: pd.DataFrame):
    # Decide if we want to color for actionSpace, NPPAutomation or Scenario
    df = df.fillna("None")
    result = []
    for idx, item in enumerate(["scenario", "automation_wrapper", "action_wrapper", "alg"]):
        fig, ax = plt.subplots(figsize=(5, 6), dpi=300, constrained_layout=True)
        groups = df.groupby(item, dropna=False)
        counter = 0
        labels = []
        for name, group in groups:
            ax.plot(
                group["return_std"],
                group["return_max"],
                marker="o",
                linestyle="",
                label=name,
                color=color_mapping[name],
            )
            ax.set_xscale("symlog")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Standardabweichung")
            ax.set_ylabel("Max Return")
            if item == "action_wrapper":
                if name == "None":
                    labels.append("ActionSpaceOption1")
                else:
                    labels.append(name[:-7])
            else:
                labels.append(name)

            counter += 1
            if item == "scenario":
                ax.legend(["Szenario 1", "Szenario 2", "Szenario 3"])
            else:
                ax.legend(groups.groups.keys())
        result.append(fig)
    plt.show()
    return result


def create_phase_2_counts_plots(df: pd.DataFrame) -> None:
    df = df.fillna("None")
    df.loc[df["action_wrapper"] == "None", "action_wrapper"] = "ActionSpaceOption1Wrapper"
    fig = plt.figure(constrained_layout=True, dpi=300)
    widths = [3, 5.5, 3]
    gs = fig.add_gridspec(1, 3, hspace=0, wspace=0, width_ratios=widths)
    ax = gs.subplots(
        sharex="col",
        sharey="row",
    )
    for idx, item in enumerate(["scenario", "obs_wrapper", "action_wrapper"]):
        groups = df.groupby(item, dropna=False)
        labels = []
        for name in list(groups.groups.keys()):
            digit = re.findall("\d+", name)
            if digit:
                labels.append(digit[0])
            else:
                labels.append((1))
        x = np.arange(len(labels))
        width = 0.45
        counts_wo_automation = [len(group.query("automation_wrapper == 'None'")) for name, group in groups]
        counts_w_automation = [
            len(group.query("automation_wrapper == 'NPPAutomationWrapper'")) for name, group in groups
        ]
        for name, group in groups:
            ax[idx].bar(x - width / 2, counts_wo_automation, width=width, label="Without Automation", color="#F65E5D")
            ax[idx].bar_label(ax[idx].containers[0], label_type="edge")
            ax[idx].bar(x + width / 2, counts_w_automation, width=width, label="With Automation", color="#1D2D5F")
            ax[idx].bar_label(ax[idx].containers[1], label_type="edge")
            ax[idx].set_ylabel("Anzahl erfolgreicher Kombinationen")
            ax[idx].set_xlabel(parse_category(item))
            ax[idx].set_xticks(x, labels)

        for axs in ax.flat:
            axs.label_outer()
    ax[1].legend(["NPPAutomation deaktiviert", "NPPAutomation aktiviert"], loc="upper left")

    plt.show()
    return fig
