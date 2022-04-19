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

            cum_reward, criticality_score, total_timesteps = evaluate(scenario, path, alg, wrapper_maker)
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
            result_dict["cum_reward"] = cum_reward
            result_dict["criticality_score"] = criticality_score
            result_dict["total_timesteps"] = total_timesteps

            df = pd.concat([df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)

        df.to_csv("output/phase_1_all_combinations_and_runs_with_returns.csv")

    # Fill NaN / None / Null values with correct labels
    df.loc[df["action_wrapper"] == "None", "action_wrapper"] = "ActionSpaceOption1Wrapper"
    df.loc[df["obs_wrapper"] == "None", "obs_wrapper"] = "ObservationOption1Wrapper"
    df.loc[df["action_wrapper"].isna(), "action_wrapper"] = "ActionSpaceOption1Wrapper"
    df.loc[df["obs_wrapper"].isna(), "obs_wrapper"] = "ObservationOption1Wrapper"

    df_wo_automation = df.query("automation_wrapper.isna() == True")
    df_w_automation = df[df["automation_wrapper"] == "NPPAutomationWrapper"]

    # Get Mean, STD and IQR for each modelling aspect:
    for item in ["obs_wrapper", "scenario", "action_wrapper", "alg"]:
        df_special = (
            df.groupby(item)
            .agg(
                return_mean=("cum_reward", "mean"),
                return_max=("cum_reward", "max"),
                return_std=("cum_reward", "std"),
                return_iqr=("cum_reward", iqr),
                # criticality_score_max=("criticality_score", "max"),
                # criticality_score_std=("criticality_score", "std"),
                # criticality_score_iqr=("criticality_score", iqr),
                # criticality_score_mean=("criticality_score", "mean"),
            )
            .round(2)
        )
        print(f"Without automation: {item}: {df_special}")
        print("-----------------Latex-------------------")
        print(df_special.to_latex())

    print(
        f"Max criticality_score without automation: {df_wo_automation.query('criticality_score==criticality_score.max()')['criticality_score']}"
    )

    highest_return_wo_automation = df_wo_automation.query("cum_reward == cum_reward.max()")
    highest_return_w_automation = df_w_automation.query("cum_reward == cum_reward.max()")

    print(
        f"Highest return without automation: {highest_return_wo_automation['cum_reward'].values[0]} from alg {highest_return_wo_automation.full_path.values[0]}"
    )
    print(
        f"Highest return with automation: {highest_return_w_automation['cum_reward'].values[0]} from alg {highest_return_w_automation.full_path.values[0]}"
    )

    statistics_wo = (
        df_wo_automation.set_index("full_path")
        .groupby("combination")
        .agg(
            return_mean=("cum_reward", "mean"),
            return_max=("cum_reward", "max"),
            return_std=("cum_reward", "std"),
            return_iqr=("cum_reward", iqr),
            timesteps_min=("total_timesteps", "min"),
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
            return_mean=("cum_reward", "mean"),
            return_max=("cum_reward", "max"),
            return_std=("cum_reward", "std"),
            return_iqr=("cum_reward", iqr),
            timesteps_min=("total_timesteps", "min"),
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

    create_multi_object_plot(statistics_wo.merge(statistics_w, how="outer"))

    paths_that_fulfil_condition_wo_automation = statistics_wo.query(
        "return_max>200 and return_std<15 and timesteps_min==250 "
    )
    paths_that_fulfil_condition_w_automation = statistics_w.query(
        "return_max>200 and return_std<15 and timesteps_min==250"
    )

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
        "scenario=='scenario1' and action_wrapper=='ActionSpaceOption3Wrapper' and timesteps_min==250"
    )
    wo_automation_scenario2_action_space3 = paths_that_fulfil_condition_wo_automation.query(
        "scenario=='scenario2' and action_wrapper=='ActionSpaceOption3Wrapper' and timesteps_min==250"
    )
    wo_automation_scenario3_action_space3 = paths_that_fulfil_condition_wo_automation.query(
        "scenario=='scenario3' and action_wrapper=='ActionSpaceOption3Wrapper' and timesteps_min==250"
    )

    w_automation_scenario1_action_space3 = paths_that_fulfil_condition_w_automation.query(
        "scenario=='scenario1' and action_wrapper=='ActionSpaceOption3Wrapper' and timesteps_min==250"
    )
    w_automation_scenario2_action_space3 = paths_that_fulfil_condition_w_automation.query(
        "scenario=='scenario2' and action_wrapper=='ActionSpaceOption3Wrapper' and timesteps_min==250"
    )
    w_automation_scenario3_action_space3 = paths_that_fulfil_condition_w_automation.query(
        "scenario=='scenario3' and action_wrapper=='ActionSpaceOption3Wrapper' and timesteps_min==250"
    )

    print(
        f"Only ActionSpace3 and scenario1 without automation:\n {wo_automation_scenario1_action_space3[['return_max','timesteps_min']]}"
    )
    print(
        f"Only ActionSpace3 and scenario2 without automation:\n {wo_automation_scenario2_action_space3[['return_max','timesteps_min']]}"
    )
    print(
        f"Only ActionSpace3 and scenario3 without automation:\n {wo_automation_scenario3_action_space3[['return_max','timesteps_min']]}"
    )

    print(
        f"Only ActionSpace3 and scenario1 with automation:\n {w_automation_scenario1_action_space3[['return_max','timesteps_min',]]}"
    )
    print(
        f"Only ActionSpace3 and scenario2 with automation:\n {w_automation_scenario2_action_space3[['return_max','timesteps_min',]]}"
    )
    print(
        f"Only ActionSpace3 and scenario3 with automation:\n {w_automation_scenario3_action_space3[['return_max','timesteps_min',]]}"
    )

    # Find best Scenario 2 with Action Space 3
    statistics_wo.query("scenario=='scenario2' and action_wrapper=='ActionSpaceOption3Wrapper'")


pd.options.display.max_colwidth = 300
df = pd.DataFrame()
try:
    df = pd.read_csv("output/phase_1_all_combinations_and_runs_with_returns.csv")
except:
    pass

start_phase_2_evaluation(df)
