import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import re

# Used for showing that random seeds have a significant influence on the result
from src.main.rl.utils.utils import parse_category


def create_multi_object_plot(df: pd.DataFrame) -> None:
    # Decide if we want to color for actionSpace, NPPAutomation or Scenario
    df = df.fillna("None")

    for item in ["scenario", "automation_wrapper", "action_wrapper"]:
        fig, ax = plt.subplots()
        groups = df.groupby(item, dropna=False)
        labels = []
        for name, group in groups:
            ax.plot(group["reward_std"], group["reward_max"], marker="o", linestyle="", label=name)
            ax.set_xscale("symlog", linthresh=10)
            # TODO decide scale
            # ax.set_xscale("log")
            # ax.set_xticks([1,10, 100])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Standardabweichung")
            ax.set_ylabel("Maximaler Reward")
            if item == "action_wrapper":
                if name == "None":
                    labels.append("ActionSpaceOption1")
                else:
                    labels.append(name[:-7])
            else:
                labels.append(name)
        ax.legend(labels)
    plt.show()


def create_phase_1_counts_plots(df: pd.DataFrame) -> None:
    df = df.fillna("None")
    for item in ["scenario", "obs_wrapper", "action_wrapper"]:
        fig, ax = plt.subplots()
        groups = df.groupby(item, dropna=False)
        labels = []
        for name in list(groups.groups.keys()):
            digit = re.findall("\d+", name)
            if digit:
                labels.append(digit[0])
            else:
                labels.append(1)
        x = np.arange(len(labels))
        width = 0.35
        counts_wo_automation = [len(group.query("automation_wrapper == 'None'")) for name, group in groups]
        counts_w_automation = [
            len(group.query("automation_wrapper == 'NPPAutomationWrapper'")) for name, group in groups
        ]
        for name, group in groups:
            ax.bar(x - width / 2, counts_wo_automation, width=width, label="Without Automation", color="red")
            ax.bar(x + width / 2, counts_w_automation, width=width, label="With Automation", color="blue")
            ax.set_ylabel("Count")
            ax.set_xlabel(parse_category(item))
            ax.set_xticks(x, labels)
            ax.legend(["Without Automation", "With Automation"])
        plt.show()
        plt.clf()
