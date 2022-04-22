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


def create_evaluation_df_phase3(path_to_save: str, paths_best_models: list = None) -> pd.DataFrame:
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
    if not paths_best_models:
        paths_best_models = [
            "../models/scenario1/training_04_06/scenario1_ActionSpaceOption3Wrapper_ObservationOption4Wrapper_None_RewardOption2Wrapper_TD3_training_04_06",
            "../models/scenario1/training_04_06/scenario1_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_SAC_training_04_06",
            "../models/scenario2/training_04_06/scenario2_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_None_RewardOption2Wrapper_PPO_training_04_06",
            "../models/scenario2/training_04_06/scenario2_ActionSpaceOption3Wrapper_ObservationOption4Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_PPO_training_04_06",
            "../models/scenario3/training_04_06/scenario3_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_None_RewardOption2Wrapper_PPO_training_04_06",
            "../models/scenario3/training_04_06/scenario3_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_A2C_training_04_06",
        ]

    df = pd.DataFrame()
    # Use Noise Wrappers
    for path in paths_best_models:
        for number in range(1, 11):
            result_dict = {}
            full_path = path + f"_{number}"
            path_to_overhand = full_path + "/best_model.zip"

            action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper = parse_wrapper(full_path)
            scenario, alg, wrapper_maker = parse_information_from_path(full_path)
            cum_reward, criticality_score, total_timesteps, action_taken, obs_taken, info = evaluate(
                scenario, path_to_overhand, alg, wrapper_maker
            )
            result_dict["combination"] = path
            result_dict["scenario"] = scenario
            result_dict["alg"] = alg
            result_dict["condensator_pump_blown"] = info[0]["Condensator_Pump Blown"]
            result_dict["water_pump_blown"] = info[0]["Water_Pump Blown"]

            result_dict["action_wrapper"] = action_wrapper.__name__ if action_wrapper else None
            result_dict["obs_wrapper"] = obs_wrapper.__name__ if obs_wrapper else None
            result_dict["automation_wrapper"] = automation_wrapper.__name__ if automation_wrapper else None
            result_dict["cum_reward"] = cum_reward
            result_dict["criticality_score"] = criticality_score
            result_dict["total_timesteps"] = total_timesteps
            result_dict["full_path"] = full_path

            # Deactivate NPPAutomationWrapper if it was activated in this specific model, else activate
            # to see how good the models perform with the different setting for NPPAutomation
            if automation_wrapper:
                wrapper_maker = WrapperMaker(action_wrapper, None, obs_wrapper, reward_wrapper)
                result_wo_automation_normal = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict["result_w_npp_automation"] = None
                result_dict["result_wo_npp_automation"] = result_wo_automation_normal[0]
                result_dict["result_w_npp_automation_criticality"] = None
                result_dict["result_wo_npp_automation_criticality"] = result_wo_automation_normal[1]
                result_dict["result_w_npp_automation_timesteps"] = None
                result_dict["result_wo_npp_automation_timesteps"] = result_wo_automation_normal[2]
            else:
                wrapper_maker = WrapperMaker(action_wrapper, NPPAutomationWrapper, obs_wrapper, reward_wrapper)
                result_w_automation_normal = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict["result_wo_npp_automation"] = None
                result_dict["result_w_npp_automation"] = result_w_automation_normal[0]
                result_dict["result_wo_npp_automation_criticality"] = None
                result_dict["result_w_npp_automation_criticality"] = result_w_automation_normal[1]
                result_dict["result_wo_npp_automation_timesteps"] = None
                result_dict["result_w_npp_automation_timesteps"] = result_w_automation_normal[2]
            # Use length == 1000
            wrapper_maker = WrapperMaker(action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper)
            result_1000_episode = evaluate(scenario, path_to_overhand, alg, wrapper_maker, None, episode_length=1000)
            result_dict["episode_length_1000_condensator_pump_blown"] = result_1000_episode[5][0][
                "Condensator_Pump Blown"
            ]
            result_dict["episode_length_1000_water_pump_blown"] = result_1000_episode[5][0]["Water_Pump Blown"]
            result_dict["episode_length_1000"] = result_1000_episode[0]
            result_dict["episode_length_1000_criticality"] = result_1000_episode[1]
            result_dict["episode_length_1000_timesteps"] = result_1000_episode[2]
            # Use Noise in evaluation
            for obs_varies_wrapper in ALL_OBS_NOISE_WRAPPERS:
                wrapper_maker = WrapperMaker(
                    action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper, None, obs_varies_wrapper
                )
                # TODO Might need to execute X times if we use randomness!
                result = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict[obs_varies_wrapper.__name__] = result[0]
                result_dict[obs_varies_wrapper.__name__ + "_criticality"] = result[1]
                result_dict[obs_varies_wrapper.__name__ + "_timesteps"] = result[2]
            for delay_wrapper in ALL_DELAY_NOISE_WRAPPERS:
                wrapper_maker = WrapperMaker(
                    action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper, delay_wrapper, None
                )
                # TODO Might need to execute X times if we use randomness!
                result = evaluate(scenario, path_to_overhand, alg, wrapper_maker)
                result_dict[delay_wrapper.__name__] = result[0]
                result_dict[delay_wrapper.__name__ + "_criticality"] = result[1]
                result_dict[delay_wrapper.__name__ + "_timesteps"] = result[2]
            # Use different starting states
            for starting_state in STARTING_STATE_OPTION1 + STARTING_STATE_OPTION2 + STARTING_STATE_OPTION3:
                wrapper_maker = WrapperMaker(action_wrapper, automation_wrapper, obs_wrapper, reward_wrapper)
                result = evaluate(scenario, path_to_overhand, alg, wrapper_maker, starting_state=starting_state())
                result_dict[starting_state.__name__] = result[0]
                result_dict[starting_state.__name__ + "_criticality"] = result[1]
                result_dict[starting_state.__name__ + "_timesteps"] = result[2]

            # add to df
            df = pd.concat([df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)
            # Fill NaN / None / Null values with correct labels
    df.loc[df["action_wrapper"] == "None", "action_wrapper"] = "ActionSpaceOption1Wrapper"
    df.loc[df["obs_wrapper"] == "None", "obs_wrapper"] = "ObservationOption1Wrapper"
    df.loc[df["action_wrapper"].isna(), "action_wrapper"] = "ActionSpaceOption1Wrapper"
    df.loc[df["obs_wrapper"].isna(), "obs_wrapper"] = "ObservationOption1Wrapper"
    df.to_csv(path_to_save)
    print(df)
    return df
