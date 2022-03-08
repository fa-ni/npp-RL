import subprocess
from time import sleep

from gym import register
from py4j.java_gateway import JavaGateway
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.parser import parse_wrapper, parse_scenario_name, parse_alg_name
from src.main.rl.utils.utils import delete_env_id, WrapperMaker, get_real_value


def eval_frontend(scenario_name: str, path: str, alg: OnPolicyAlgorithm, wrapper: WrapperMaker):
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=wrapper.make_wrapper)
    model = alg.load(path)
    p = subprocess.Popen(["java", "-jar", "npp-automation.jar"])
    while True:
        try:
            gateway = JavaGateway(java_process=p)  # connect to the JVM
            entry = gateway.entry_point
            frontend = entry.getNPPUI()
            obs = vec_env.reset()
            parsed_scenario_name = parse_scenario_name(scenario_name)
            for i in range(250):
                action, _states = model.predict(obs)
                action_size = len(action[0])
                if parsed_scenario_name == "scenario1":
                    moderator_setting = get_real_value(100, action[0][0])
                    wp1rpm_setting = get_real_value(2000, action[0][1])
                    frontend.getSliderWP1RPM().setValue(wp1rpm_setting)
                    frontend.getSliderRodPos().setValue(moderator_setting)
                    if action_size >= 3:
                        frontend.getBCLWV1().doClick() if int(action[0][2]) == 0 else frontend.getBOpWV1().doClick()
                    if action_size == 5:
                        frontend.getBCLFV1().doClick() if int(action[0][3]) == 0 else frontend.getBOpFV1().doClick()
                        cprpm_setting = get_real_value(2000, action[0][4])
                        frontend.getSliderCPRPM().setValue(cprpm_setting)
                elif parsed_scenario_name == "scenario2":
                    moderator_setting = -1 if action[0][0] == 0 else 1
                    wp1rpm_setting = -25 if action[0][1] == 0 else +25
                    frontend.getSliderWP1RPM().setValue(frontend.getSliderWP1RPM().getValue() + wp1rpm_setting)
                    frontend.getSliderRodPos().setValue(frontend.getSliderRodPos().getValue() + moderator_setting)
                if action_size >= 3:
                    frontend.getBCLWV1().doClick() if int(action[0][2]) == 0 else frontend.getBOpWV1().doClick()
                if action_size == 5:
                    frontend.getBCLFV1().doClick() if int(action[0][3]) == 0 else frontend.getBOpFV1().doClick()
                    cprpm_setting = -25 if action[0][4] == 0 else +25
                    frontend.getSliderCPRPM().setValue(frontend.getSliderCPRPM().getValue() + cprpm_setting)
                # TODO if needed scenario3
                frontend.timeStep()
                # This is needed to get the same state for the environment. This is not the best way to do it.
                # Other option is to create a new env where only the fronend-calls are used, so we do not need
                # to do this here.
                obs, reward, done, info = vec_env.step(action)
                if done:
                    print("Done")
                    break
            gateway.shutdown()
            p.kill()
            break
        except:
            pass


path = "scenario1ObservationOption5WrapperActionOption3WrapperRewardOption2WrapperSAC/best_model0.zip"
scenario, alg, wrapper_maker = parse_information_from_path(path)
eval_frontend(scenario, path, alg, wrapper_maker)
