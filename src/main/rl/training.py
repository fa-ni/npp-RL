from pathlib import Path

from gym import register
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from src.main.rl.utils.constants import ALL_ACTION_WRAPPERS, ALL_OBSERVATION_WRAPPERS
from src.main.rl.utils.utils import WrapperMaker, parse_scenario_name, delete_env_id
from src.main.rl.wrapper.npp_automation_wrapper import NPPAutomationWrapper
from src.main.rl.wrapper.reward_wrapper3 import RewardOption3Wrapper

log_dir = "./model"


def train_agent(
    algorithm,
    environment,
    scenario_name: str,
    action_wrapper_name: str = None,
    obs_wrapper_name: str = None,
    npp_automation: str = None,
    reward_wrapper_name: str = None,
    name_ending: str = None,
):
    try:
        log_name_scenario = parse_scenario_name(scenario_name)
        log_name = f"{log_name_scenario}_{action_wrapper_name}_{obs_wrapper_name}_{npp_automation}_{reward_wrapper_name}_{algorithm.__name__}_{name_ending}"
        environment.reset()
        # This is used to use the same logic for model saving as for logs to directly identify them
        continue_search = True
        counter = 1
        model_save_path = ""
        while continue_search:
            my_file = Path(f"./models/{log_name_scenario}/{name_ending}/{log_name}_{counter}")
            if not my_file.is_dir():
                continue_search = False
                model_save_path = f"./models/{log_name_scenario}/{name_ending}/{log_name}_{counter}"
            counter += 1
        eval_callback = EvalCallback(environment, eval_freq=3000, best_model_save_path=model_save_path, verbose=1)

        model = algorithm(
            "MlpPolicy",
            env=environment,
            verbose=1,
            tensorboard_log=f"./logs/{log_name_scenario}/{name_ending}/",
            device="cpu",
        ).learn(
            100000,
            tb_log_name=log_name,
            callback=eval_callback,
        )
    except Exception as exception:
        print("Error")
        print(exception)


def train_all_scenarios(
    scenarios: list,
    reward_wrapper,
    name_ending: str = None,
):
    env_id = f"env-v1"
    # Algorithms
    algorithms = [A2C, PPO, TD3, SAC]
    # Whether to run with or without NPPAutomation active
    npp_automation = [None, NPPAutomationWrapper]
    for scenario in scenarios:
        parsed_scenario_name = parse_scenario_name(scenario)
        for npp_automation_wrapper in npp_automation:
            # With Wrappers
            for action_wrapper in ALL_ACTION_WRAPPERS:
                for observation_wrapper in ALL_OBSERVATION_WRAPPERS:
                    for alg in algorithms:
                        if alg == TD3:
                            num_cpu = 1
                        else:
                            num_cpu = 8
                        register(id=env_id, entry_point=scenario)
                        # This is needed because make_vec_env does not allow a method with multiple parameters as wrapper_class
                        wrapper_maker = WrapperMaker(
                            action_wrapper, observation_wrapper, npp_automation_wrapper, reward_wrapper
                        )
                        vec_env = make_vec_env(
                            env_id, n_envs=num_cpu, wrapper_class=wrapper_maker.make_wrapper, monitor_dir=log_dir
                        )
                        vec_env_monitor = VecMonitor(vec_env)

                        train_agent(
                            alg,
                            vec_env_monitor,
                            scenario,
                            action_wrapper.__name__,
                            observation_wrapper.__name__,
                            npp_automation_wrapper.__name__ if npp_automation_wrapper != None else None,
                            reward_wrapper.__name__ if reward_wrapper != None else None,
                            name_ending,
                        )
                        delete_env_id(env_id)
        # Only NPPAutomationWrapper/No Wrapper with ActionWrapper
        for npp_automation_wrapper in npp_automation:
            for action_wrapper in ALL_ACTION_WRAPPERS:
                for alg in algorithms:
                    if alg == TD3:
                        num_cpu = 1
                    else:
                        num_cpu = 8
                    register(id=env_id, entry_point=scenario)
                    wrapper_maker = WrapperMaker(action_wrapper, None, NPPAutomationWrapper, reward_wrapper)
                    vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=wrapper_maker.make_wrapper)
                    vec_env_monitor = VecMonitor(vec_env)
                    train_agent(
                        alg,
                        vec_env_monitor,
                        scenario,
                        action_wrapper.__name__,
                        None,
                        npp_automation_wrapper.__name__ if npp_automation_wrapper != None else None,
                        reward_wrapper.__name__ if reward_wrapper != None else None,
                        name_ending,
                    )
                    delete_env_id(env_id)
            # Only NPPAutomationWrapper/No Wrapper with ObsWrapper
            for observation_wrapper in ALL_OBSERVATION_WRAPPERS:
                for alg in algorithms:
                    if alg == TD3:
                        num_cpu = 1
                    else:
                        num_cpu = 8
                    register(id=env_id, entry_point=scenario)
                    wrapper_maker = WrapperMaker(None, observation_wrapper, NPPAutomationWrapper, reward_wrapper)
                    vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=wrapper_maker.make_wrapper)
                    vec_env_monitor = VecMonitor(vec_env)
                    train_agent(
                        alg,
                        vec_env_monitor,
                        scenario,
                        None,
                        observation_wrapper.__name__,
                        npp_automation_wrapper.__name__ if npp_automation_wrapper != None else None,
                        reward_wrapper.__name__ if reward_wrapper != None else None,
                        name_ending,
                    )
                    delete_env_id(env_id)
        # Only NPPAutomationWrapper/No Wrapper
        for npp_automation_wrapper in npp_automation:
            for alg in algorithms:
                if alg == TD3:
                    num_cpu = 1
                else:
                    num_cpu = 8
                register(id=env_id, entry_point=scenario)
                vec_env = make_vec_env(env_id, n_envs=num_cpu, wrapper_class=RewardOption3Wrapper, monitor_dir=log_dir)
                vec_env_monitor = VecMonitor(vec_env)
                train_agent(
                    alg,
                    vec_env_monitor,
                    scenario,
                    None,
                    None,
                    npp_automation_wrapper.__name__ if npp_automation_wrapper != None else None,
                    reward_wrapper.__name__ if reward_wrapper != None else None,
                    name_ending,
                )
                delete_env_id(env_id)
