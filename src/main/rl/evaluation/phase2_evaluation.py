import glob

import pandas as pd
from scipy.stats import iqr

from src.main.rl.evaluation.eval import evaluate
from src.main.rl.evaluation.plots.phase_2_plots import create_multi_object_plot, create_phase_2_counts_plots
from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.parser import parse_wrapper, parse_scenario_name


def start_phase_2_evaluation(df: pd.DataFrame = pd.DataFrame()):
    if df.empty:
        all_files = []
        # TODO change path
        for file in glob.glob("../models/*/*/*/*.zip", recursive=True):
            all_files.append(file)
        # Check if all models have been found
        assert len(all_files) == 2400

        pd.options.display.max_colwidth = 200
        for path in all_files:
            result_dict = {}
            scenario, alg, wrapper_maker = parse_information_from_path(path)
            action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper = parse_wrapper(path)

            reward, criticality_score = evaluate(scenario, path, alg, wrapper_maker)
            result_dict["full_path"] = path
            combination_name = path[:-17]
            if combination_name.endswith("_"):
                result_dict["combination"] = combination_name[:-1]
            else:
                result_dict["combination"] = combination_name
            result_dict["scenario"] = parse_scenario_name(path)
            result_dict["alg"] = alg
            result_dict["action_wrapper"] = action_wrapper.__name__ if action_wrapper else None
            result_dict["obs_wrapper"] = obs_wrapper.__name__ if obs_wrapper else None
            result_dict["automation_wrapper"] = automation_wrapper.__name__ if automation_wrapper else None
            result_dict["reward"] = reward
            result_dict["criticality_score"] = criticality_score

            df = pd.concat([df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)

        df.to_csv("output/phase_1_all_combinations_and_runs_with_rewards.csv")

    df_w_automation = df[df["automation_wrapper"] == "NPPAutomationWrapper"]
    df_wo_automation = df.query("automation_wrapper.isna() == True")

    highest_reward_wo_automation = df_wo_automation.query("reward == reward.max()")
    highest_reward_w_automation = df_w_automation.query("reward == reward.max()")

    print(
        f"Highest reward without automation: {highest_reward_wo_automation.reward.values[0]} from alg {highest_reward_wo_automation.full_path.values[0]}"
    )
    print(
        f"Highest reward with automation: {highest_reward_w_automation.reward.values[0]} from alg {highest_reward_w_automation.full_path.values[0]}"
    )

    statistics_wo = (
        df_wo_automation.set_index("full_path")
        .groupby("combination")
        .agg(
            reward_max=("reward", "max"),
            reward_std=("reward", "std"),
            reward_iqr=("reward", iqr),
            criticality_score_max=("criticality_score", "max"),
            criticality_score_std=("criticality_score", "std"),
            criticality_score__iqr=("criticality_score", iqr),
            scenario=("scenario", "first"),
            alg=("alg", "first"),
            action_wrapper=("action_wrapper", "first"),
            obs_wrapper=("obs_wrapper", "first"),
            automation_wrapper=("automation_wrapper", "first"),
        )
    )

    statistics_w = (
        df_w_automation.set_index("full_path")
        .groupby("combination")
        .agg(
            reward_max=("reward", "max"),
            reward_std=("reward", "std"),
            criticality_score_max=("criticality_score", "max"),
            criticality_score_std=("criticality_score", "std"),
            scenario=("scenario", "first"),
            alg=("alg", "first"),
            action_wrapper=("action_wrapper", "first"),
            obs_wrapper=("obs_wrapper", "first"),
            automation_wrapper=("automation_wrapper", "first"),
        )
    )

    create_multi_object_plot(statistics_wo.merge(statistics_w, how="outer"))

    paths_that_fulfil_condition_wo_automation = statistics_wo.query("reward_max>200 and reward_std<15")
    paths_that_fulfil_condition_w_automation = statistics_w.query("reward_max>200 and reward_std<15")

    create_phase_2_counts_plots(
        paths_that_fulfil_condition_wo_automation.merge(paths_that_fulfil_condition_w_automation, how="outer")
    )

    print(f"All paths that fulfill the condition without Automation: {paths_that_fulfil_condition_wo_automation}")
    print(f"All paths that fulfill the condition with Automation: {paths_that_fulfil_condition_w_automation}")

    print(f"Number of items with condition without Automation: {len(paths_that_fulfil_condition_wo_automation)}")
    print(f"Number of items with condition with Automation: {len(paths_that_fulfil_condition_w_automation)}")

    cols_to_count = ["alg", "scenario", "action_wrapper", "obs_wrapper", "automation_wrapper"]
    statistics_wo_value_counts = pd.Series()
    statistics_w_value_counts = pd.Series()
    for col in cols_to_count:
        statistics_wo_value_counts = pd.concat(
            [statistics_wo_value_counts, paths_that_fulfil_condition_wo_automation[col].value_counts()]
        )
        statistics_w_value_counts = pd.concat(
            [statistics_w_value_counts, paths_that_fulfil_condition_w_automation[col].value_counts()]
        )

    print(f"Counts without Automation: {statistics_wo_value_counts}")
    print(f"Counts withAutomation: {statistics_w_value_counts}")

    # Only ActionSpaceOption3Wrapper:
    wo_automation_scenario1_action_space3 = paths_that_fulfil_condition_wo_automation.query(
        "scenario=='scenario1' and action_wrapper=='ActionSpaceOption3Wrapper'"
    )
    wo_automation_scenario2_action_space3 = paths_that_fulfil_condition_wo_automation.query(
        "scenario=='scenario2' and action_wrapper=='ActionSpaceOption3Wrapper'"
    )
    wo_automation_scenario3_action_space3 = paths_that_fulfil_condition_wo_automation.query(
        "scenario=='scenario3' and action_wrapper=='ActionSpaceOption3Wrapper'"
    )

    w_automation_scenario1_action_space3 = paths_that_fulfil_condition_w_automation.query(
        "scenario=='scenario1' and action_wrapper=='ActionSpaceOption3Wrapper'"
    )
    w_automation_scenario2_action_space3 = paths_that_fulfil_condition_w_automation.query(
        "scenario=='scenario2' and action_wrapper=='ActionSpaceOption3Wrapper'"
    )
    w_automation_scenario3_action_space3 = paths_that_fulfil_condition_w_automation.query(
        "scenario=='scenario3' and action_wrapper=='ActionSpaceOption3Wrapper'"
    )

    print(f"Only ActionSpace3 and scenario1 without automation:\n {wo_automation_scenario1_action_space3}")
    print(f"Only ActionSpace3 and scenario2 without automation:\n {wo_automation_scenario2_action_space3}")
    print(f"Only ActionSpace3 and scenario3 without automation:\n {wo_automation_scenario3_action_space3}")

    print(f"Only ActionSpace3 and scenario1 with automation:\n {w_automation_scenario1_action_space3}")
    print(f"Only ActionSpace3 and scenario2 with automation:\n {w_automation_scenario2_action_space3}")
    print(f"Only ActionSpace3 and scenario3 with automation:\n {w_automation_scenario3_action_space3}")


pd.options.display.max_colwidth = 150
df = pd.DataFrame()
try:
    df = pd.read_csv("output/phase_1_all_combinations_and_runs_with_rewards.csv")
except:
    pass
start_phase_2_evaluation(df)
