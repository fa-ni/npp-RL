import subprocess

from gym import register
from py4j.java_gateway import JavaGateway
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.constants import (
    scaling_factors_scenario_1,
    scaling_factors_scenario_2,
    scaling_factors_scenario_3,
)
from src.main.rl.utils.parser import parse_scenario_name
from src.main.rl.utils.utils import delete_env_id, WrapperMaker
from src.main.rl.wrapper.reward_calculations import calculate_roofed_reward


def eval_frontend(scenario_name: str, path: str, alg: OnPolicyAlgorithm, wrappers: WrapperMaker) -> None:
    """
    This function is used to use an agent with the java-frontend to visualize the actions taken by the agent.
    Only ActionSpaceOption3 agents are currently supported.
    The jar files are within this project which can be used.
    The trained agent is predicting actions, which are then set via a python-java bridge in the java GUI.
    The java-backend then get triggered from here to execute the logic with the values from the GUI.
    """
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=wrappers.make_wrapper)
    model = alg.load(path)
    if wrappers.npp_automation_wrapper:
        p = subprocess.Popen(["java", "-jar", "npp_with_automation.jar"])
    else:
        p = subprocess.Popen(["java", "-jar", "npp_without_automation.jar"])
    # sleep(1)
    while True:
        try:
            y = []

            gateway = JavaGateway(java_process=p)  # connect to the JVM
            entry = gateway.entry_point
            frontend = entry.getNPPUI()
            obs = vec_env.reset()
            info = [{"ModPercent": 100}]
            mod_change_with_automation = False
            parsed_scenario_name = parse_scenario_name(scenario_name)
            for i in range(250):
                scaled_actions = []

                action, _states = model.predict(obs, deterministic=True)
                if parsed_scenario_name == "scenario1":
                    for idx, item in enumerate(action[0]):
                        scaled_actions.append(int(((item + 1) * (scaling_factors_scenario_1[idx] / 2))))
                    # This is needed as the UI will not send a change event if nothing changes. The model
                    # always sets the new value (does not matter whether it changes or not). The issue is that
                    # the poisoning factor is changing with the number of value setting. The value from the
                    # Moderator Percentage / Control Rods will not change. This will only update the modePer with
                    # the same value and therefore will also change the poisoning factor as desired.
                    if scaled_actions[0] == 100 - frontend.getModPer():
                        frontend.fireChange()
                    frontend.getSliderRodPos().setValue(scaled_actions[0])
                    frontend.getSliderWP1RPM().setValue(scaled_actions[1])
                    frontend.getBCLWV1().doClick() if action[0][2] < 0 else frontend.getBOpWV1().doClick()
                    frontend.getBCLFV1().doClick() if action[0][3] < 0 else frontend.getBOpFV1().doClick()
                    frontend.getSliderCPRPM().setValue(scaled_actions[4])

                elif parsed_scenario_name == "scenario2":
                    for idx, item in enumerate(action[0]):
                        scaled_actions.append(
                            -scaling_factors_scenario_2[idx] if item == 0 else scaling_factors_scenario_2[idx]
                        )
                    frontend.getSliderRodPos().setValue(frontend.getSliderRodPos().getValue() + scaled_actions[0])
                    frontend.getSliderWP1RPM().setValue(frontend.getSliderWP1RPM().getValue() + scaled_actions[1])
                    frontend.getBCLWV1().doClick() if action[0][2] == 0 else frontend.getBOpWV1().doClick()
                    frontend.getBCLFV1().doClick() if action[0][3] == 0 else frontend.getBOpFV1().doClick()
                    frontend.getSliderCPRPM().setValue(frontend.getSliderCPRPM().getValue() + scaled_actions[4])

                elif parsed_scenario_name == "scenario3":
                    for idx, item in enumerate(action[0]):
                        scaled_actions.append(scaling_factors_scenario_3[idx][item])
                    y.append(scaled_actions)
                    if scaled_actions[0] == 0 or (scaled_actions[0] < 0 and frontend.getModPer() == 100):
                        frontend.fireChange()
                    frontend.getSliderRodPos().setValue(frontend.getSliderRodPos().getValue() + scaled_actions[0])
                    frontend.getSliderWP1RPM().setValue(frontend.getSliderWP1RPM().getValue() + scaled_actions[1])
                    frontend.getBCLWV1().doClick() if action[0][2] == 0 else frontend.getBOpWV1().doClick()
                    frontend.getBCLFV1().doClick() if action[0][3] == 0 else frontend.getBOpFV1().doClick()
                    frontend.getSliderCPRPM().setValue(frontend.getSliderCPRPM().getValue() + scaled_actions[4])

                power = frontend.timeStep()

                # This is needed to get the same state for the environment. This is not the best way to do it.
                # Other option is to create a new env where only the frontend-calls are used, so we do not need
                # to do this here.
                obs, reward, done, info = vec_env.step(action)

                # Debug
                # if round(reward[0].astype(float), 3) != round(calculate_roofed_reward(power), 3):
                #    print("STOP")
                if done:
                    print("Done")
                    break
            gateway.shutdown()
            p.kill()
            break
        except:
            pass


# path = "../models/models/scenario3/training_04_06/scenario3_ActionSpaceOption3Wrapper_ObservationOption5Wrapper_NPPAutomationWrapper_RewardOption2Wrapper_PPO_training_04_06_1/best_model.zip"
# scenario, alg, wrapper_maker = parse_information_from_path(path)
# eval_frontend(scenario, path, alg, wrapper_maker)
