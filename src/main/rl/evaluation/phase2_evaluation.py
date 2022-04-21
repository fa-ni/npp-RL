import glob

import pandas as pd

from src.main.rl.evaluation.eval import evaluate
from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.parser import parse_wrapper, parse_scenario_name


def create_evaluation_df_phase2(path_to_save: str, all_files=None):
    df = pd.DataFrame()
    if not all_files:
        all_files = []
        for file in glob.glob("../models/*/*/*/*.zip", recursive=True):
            all_files.append(file)
        # Check if all models have been found
    assert len(all_files) == 2400

    pd.options.display.max_colwidth = 200
    for path in all_files:
        result_dict = {}
        scenario, alg, wrapper_maker = parse_information_from_path(path)
        action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper = parse_wrapper(path)

        result = evaluate(scenario, path, alg, wrapper_maker)

        cum_reward = result[0]
        criticality_score = result[1]
        total_timesteps = result[2]
        result_dict["full_path"] = path
        combination_name = path[:-17]
        # This is needed because we have numbers from 1 to 10 and if we cut the tenth combination we do not get
        # the correct value
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

        # Fill NaN / None / Null values with correct labels
    df.loc[df["action_wrapper"] == "None", "action_wrapper"] = "ActionSpaceOption1Wrapper"
    df.loc[df["obs_wrapper"] == "None", "obs_wrapper"] = "ObservationOption1Wrapper"
    df.loc[df["action_wrapper"].isna(), "action_wrapper"] = "ActionSpaceOption1Wrapper"
    df.loc[df["obs_wrapper"].isna(), "obs_wrapper"] = "ObservationOption1Wrapper"
    # Save
    df.to_csv(path_to_save)
