import glob
import statistics
from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.main.rl.evaluation.eval import evaluate
from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.parser import parse_wrapper
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper


def get_items_with_over_200_and_difference_less_than_25(
    dict_combination_to_reward: Dict[list], max_dif=25, min_reward=200
) -> list:
    paths = []
    biggest_difference_between_runs = []
    for key, values in dict_combination_to_reward.items():
        if len(values) == 10:
            values.sort()
            std_values = statistics.stdev(values)
            print(std_values)
            # TODO decide what is the best here
            # biggest_difference_between_runs.append(values[0] - values[9])
            # if abs(values[0] - values[9]) < max_dif:
            #    if values[9] > min_reward:  # TODO either all above 200 or only one?
            #        paths.append(key)
    # TODO plotting
    plt.boxplot(biggest_difference_between_runs)
    plt.title("Difference between the two runs in Reward")
    plt.show()
    return paths


def get_numbers(paths: list) -> dict:
    list = []
    for item in paths:
        scenario, alg, wrapper_maker = parse_information_from_path(item)
        wrapper = parse_wrapper(item)
        list.append(scenario)
        list.append(alg)
        list.extend(wrapper)
    return dict((Counter(list)))


def detailed_analysis_for_action_space_and_scenarios(paths: list, action_space):
    paths_with_action_space_and_scenario_1 = []
    paths_with_action_space_and_scenario_2 = []
    paths_with_action_space_and_scenario_3 = []
    for item in paths:
        scenario, alg, wrapper_maker = parse_information_from_path(item)
        wrapper = parse_wrapper(item)
        if wrapper[0] == action_space and scenario == "envs.scenario1:Scenario1":
            paths_with_action_space_and_scenario_1.append(item)
        if wrapper[0] == action_space and scenario == "envs.scenario2:Scenario2":
            paths_with_action_space_and_scenario_2.append(item)
        if wrapper[0] == action_space and scenario == "envs.scenario3:Scenario3":
            paths_with_action_space_and_scenario_3.append(item)
    return (
        paths_with_action_space_and_scenario_1,
        paths_with_action_space_and_scenario_2,
        paths_with_action_space_and_scenario_3,
    )


files_with_automation = []
files_without_automation = []
all_files = []
for file in glob.glob("models/*/*/*/*.zip", recursive=True):
    if "NPPAutomationWrapper" in file:
        files_with_automation.append(file)
    else:
        files_without_automation.append(file)
    all_files.append(file)
print(f"Length without Automation Total: {len(files_without_automation)}")
print(f"Length with Automation Total: {len(files_with_automation)}")

highest_reward_with_automation = 0
highest_reward_without_automation = 0
path_highest_reward_without_automation = ""
path_highest_reward_with_automation = ""
# All runs have a different path and a different reward
dict_path_to_reward = {}
# Here we save every run of a combination as a list in a dict
dict_combination_to_reward = {}
for item in files_without_automation:
    scenario, alg, wrapper_maker = parse_information_from_path(item)
    reward = evaluate(scenario, item, alg, wrapper_maker)[0]
    dict_path_to_reward[item] = reward
    if item[:-17] in dict_combination_to_reward.keys():
        dict_combination_to_reward[item[:-17]].append(reward)
    else:
        dict_combination_to_reward[item[:-17]] = [reward]
    if reward > highest_reward_without_automation:
        highest_reward_without_automation = reward
        path_highest_reward_without_automation = item
print(f"Highest overall reward: {highest_reward_without_automation} from alg {path_highest_reward_without_automation}")
for item in files_with_automation:
    scenario, alg, wrapper_maker = parse_information_from_path(item)
    reward = evaluate(scenario, item, alg, wrapper_maker)[0]
    dict_path_to_reward[item] = reward
    if item[:-17] in dict_combination_to_reward.keys():
        dict_combination_to_reward[item[:-17]].append(reward)
    else:
        dict_combination_to_reward[item[:-17]] = [reward]
    if reward > highest_reward_with_automation:
        highest_reward_with_automation = reward
        path_highest_reward_with_automation = item
print(f"Highest overall reward: {highest_reward_with_automation} from alg {path_highest_reward_with_automation}")
print(f"Every Combination and reward: {dict_path_to_reward}")
df = pd.DataFrame(dict_path_to_reward)
df.to_csv("/output/phase_1_all_combinations_and_runs_with_rewards.csv")

# Check if all models have been found
# assert len(files_without_automation) == 360
# assert len(files_with_automation) == 360

print("-----With Automation-----")
selected_paths_w_automation = get_items_with_over_200_and_difference_less_than_25(dict_combination_to_reward)
print(f"All paths that fulfill the condition: {selected_paths_w_automation}")
print(f"Number of items with condition: {len(selected_paths_w_automation)}")
dict_with_counts_all_w_automation = get_numbers(selected_paths_w_automation)
print(f"Dict with counts for all: {dict_with_counts_all_w_automation}")

# Only ActionSpaceOption3Wrapper:
paths_with_action_space3_w_automation = detailed_analysis_for_action_space_and_scenarios(
    selected_paths_w_automation, ActionSpaceOption3Wrapper
)

print(f"Only ActionSpace3 and scenario1: {paths_with_action_space3_w_automation[0]}")
print(f"Only ActionSpace3 and scenario2: {paths_with_action_space3_w_automation[1]}")
print(f"Only ActionSpace3 and scenario3: {paths_with_action_space3_w_automation[2]}")

# WITHOUT AUTOMATION
print("-----Without Automation-----")

selected_paths_wo_automation = get_items_with_over_200_and_difference_less_than_25(dict_combination_to_reward)
# How many of each thing (scenario,obs,action,alg)
# Note None value is the sum of all Nones from Obs and Action together. So we need to manually calculate it
# for each part.
print(f"All paths that fulfill the condition: {selected_paths_wo_automation}")
print(f"Number of items with condition: {len(selected_paths_wo_automation)}")
dict_with_counts_all_wo_automation = get_numbers(selected_paths_wo_automation)
print(f"Dict with counts for all: {dict_with_counts_all_wo_automation}")
# Only ActionSpaceOption3Wrapper:
paths_with_action_space3_wo_automation = detailed_analysis_for_action_space_and_scenarios(
    selected_paths_wo_automation, ActionSpaceOption3Wrapper
)

print(f"Only ActionSpace3 and scenario1: {paths_with_action_space3_wo_automation[0]}")
print(f"Only ActionSpace3 and scenario2: {paths_with_action_space3_wo_automation[1]}")
print(f"Only ActionSpace3 and scenario3: {paths_with_action_space3_wo_automation[2]}")
