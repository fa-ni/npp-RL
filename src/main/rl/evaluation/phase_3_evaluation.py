import numpy as np
import pandas as pd

from src.main.rl.evaluation.eval import evaluate
from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.constants import (
    ALL_OBS_NOISE_WRAPPERS,
    ALL_DELAY_NOISE_WRAPPERS,
    STARTING_STATE_OPTION1,
    STARTING_STATE_OPTION2,
    STARTING_STATE_OPTION3,
)
from src.main.rl.utils.parser import parse_wrapper
from src.main.rl.utils.utils import WrapperMaker
from src.main.rl.wrapper.npp_automation_wrapper import NPPAutomationWrapper


def create_evaluation_df() -> pd.DataFrame:
    """
    This function creates further evaluations for the best combinations.
    For each model that was trained (10 for each combinations) a full evaluation is done.
    Full evaluation means that for every of the following options a reward and a criticality score is calculated.
    Options:
    - Normal execution
    - Execution with episode length==1000
    - Execution for different noise wrappers
        - Different Observation Noise Wrappers
        - Different Delay Noise Wrappers
    - Execution for different starting states
        - Medium starting state, no critical states
        - Medium starting state, critical states
        - Artificial starting state
    More information can be found in the master thesis itself.
    """
    paths_best_models = [
        "scenario1/training_18_03/scenario1_ActionSpaceOption3Wrapper_ObservationOption4Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_TD3_training_18_03",
        "scenario1/training_18_03/scenario1_ActionSpaceOption3Wrapper_ObservationOption4Wrapper_None_RewardOption2Wrapper_TD3_training_18_03",
    ]

    df = pd.DataFrame()
    # Use Noise Wrappers
    for path in paths_best_models:
        for number in range(1, 3):
            result_dict = {}

            full_path = path + f"_{number}"
            result_dict["full_path"] = full_path
            result_dict["combination"] = path
            path_to_overhand = "../" + full_path + "/best_model.zip"
            action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper = parse_wrapper(full_path)
            scenario, alg, wrapper_maker = parse_information_from_path(full_path)
            result_normal = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
            result_dict["normal_result"] = result_normal[0]
            result_dict["normal_result_criticality"] = result_normal[1]
            # Deactivate NPPAutomationWrapper if it was activated in this specific model, else activate
            # to see how good the models perform with the different setting for NPPAutomation
            if automation_wrapper:
                wrapper_maker = WrapperMaker(action_wrapper, None, obs_wrapper, reward_wrapper)
                result_wo_automation_normal = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict["result_w_npp_automation"] = None
                result_dict["result_wo_npp_automation"] = result_wo_automation_normal[0]
                result_dict["result_w_npp_automation_criticality"] = None
                result_dict["result_wo_npp_automation_criticality"] = result_wo_automation_normal[1]
            else:
                wrapper_maker = WrapperMaker(action_wrapper, NPPAutomationWrapper, obs_wrapper, reward_wrapper)
                result_w_automation_normal = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict["result_wo_npp_automation"] = None
                result_dict["result_w_npp_automation"] = result_w_automation_normal[0]
                result_dict["result_wo_npp_automation_criticality"] = None
                result_dict["result_w_npp_automation_criticality"] = result_w_automation_normal[1]
            # Use length == 1000
            wrapper_maker = WrapperMaker(action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper)
            result_1000_episode = evaluate(scenario, path_to_overhand, alg, wrapper_maker, None, episode_length=1000)
            print(result_1000_episode)
            result_dict["episode_length_1000"] = result_1000_episode[0]
            result_dict["episode_length_1000_criticality"] = result_1000_episode[1]

            for obs_varies_wrapper in ALL_OBS_NOISE_WRAPPERS:
                wrapper_maker = WrapperMaker(
                    action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper, None, obs_varies_wrapper
                )
                # TODO Might need to execute X times if we use randomness!
                result = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict[obs_varies_wrapper.__name__] = result[0]
                result_dict[obs_varies_wrapper.__name__ + "_criticality"] = result[1]

            for delay_wrapper in ALL_DELAY_NOISE_WRAPPERS:
                wrapper_maker = WrapperMaker(
                    action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper, delay_wrapper, None
                )
                # TODO Might need to execute X times if we use randomness!
                result = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict[delay_wrapper.__name__] = result[0]
                result_dict[delay_wrapper.__name__ + "_criticality"] = result[1]

            # Use different starting states
            for starting_state in STARTING_STATE_OPTION1 + STARTING_STATE_OPTION2 + STARTING_STATE_OPTION3:
                wrapper_maker = WrapperMaker(action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper)
                result = evaluate(scenario, path_to_overhand, alg, wrapper_maker, starting_state=starting_state())
                print(result)
                result_dict[starting_state.__name__] = result[0]
                result_dict[starting_state.__name__ + "_criticality"] = result[1]

            # add to df
            df = pd.concat([df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)
            print(df)
            # Count critical state

    print(df)
    return df


result_df = create_evaluation_df()
# Create statistics about each model over all tests
reward_cols = [col for col in result_df.columns if not "_criticality" in col]
critical_cols = [col for col in result_df.columns if "_criticality" in col]
df_reward_statistics_with_columns = (
    result_df[reward_cols]
    .set_index(["full_path", "combination"])
    .agg(["min", "max", "std", "mean"], axis=1)
    .reset_index()
    .rename(columns={"min": "reward_min", "max": "reward_max", "std": "reward_std", "mean": "reward_mean"})
)
df_critical_statistics_with_columns = (
    result_df[critical_cols + ["full_path", "combination"]]
    .set_index(["full_path", "combination"])
    .agg(["min", "max", "std", "mean"], axis=1)
    .reset_index()
    .rename(columns={"min": "critical_min", "max": "critical_max", "std": "critical_std", "mean": "critical_mean"})
)

combined_statistics = pd.merge(
    df_reward_statistics_with_columns, df_critical_statistics_with_columns, how="inner", on=["full_path", "combination"]
)
print(combined_statistics)
# Create statistics over every combination and all tests

# TODO is hard this way around


# Create statistics over every combination per tests
result_df_grouped_by_models = (
    result_df.drop(columns=["full_path"]).groupby("combination").agg(["min", "max", "std", "mean"])
)
print(result_df_grouped_by_models)
