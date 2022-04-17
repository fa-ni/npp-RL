import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import re

# Used for showing that random seeds have a significant influence on the result
def create_multi_object_plot(df: pd.DataFrame) -> None:
    # Decide if we want to color for actionSpace, NPPAutomation or Scenario
    df = df.fillna("None")

    for item in ["scenario", "automation_wrapper", "action_wrapper"]:
        fig, ax = plt.subplots()
        groups = df.groupby(item, dropna=False)
        for name, group in groups:
            ax.plot(group["reward_std"], group["reward_max"], marker="o", linestyle="", label=name)

            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Standardabweichung")
            ax.set_ylabel("Max Reward")
        ax.legend(groups.groups.keys())
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
                labels.append("None")
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
            ax.set_xlabel(item)
            ax.set_xticks(x, labels)
            ax.legend(["Without Automation", "With Automation"])
        plt.show()
        plt.clf()
