import subprocess

from gym import register
from py4j.java_gateway import JavaGateway
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.main.rl.utils.combined_parser import parse_information_from_path
from src.main.rl.utils.parser import parse_scenario_name
from src.main.rl.utils.utils import delete_env_id, WrapperMaker, get_real_value


def eval_frontend(scenario_name: str, path: str, alg: OnPolicyAlgorithm, wrapper: WrapperMaker):
    env_id = "TestEnv-v1"
    delete_env_id(env_id)

    register(id=env_id, entry_point=scenario_name)
    vec_env = make_vec_env(env_id, n_envs=1, wrapper_class=wrapper.make_wrapper)
    model = alg.load(path)
    p = subprocess.Popen(["java", "-jar", "npp-java-adjusted-main.jar"])

    while True:
        try:
            gateway = JavaGateway(java_process=p)  # connect to the JVM
            entry = gateway.entry_point
            frontend = entry.getNPPUI()
            obs = vec_env.reset()
            parsed_scenario_name = parse_scenario_name(scenario_name)
            for i in range(250):
                print(i)
                action, _states = model.predict(obs, deterministic=True)
                if parsed_scenario_name == "scenario1":
                    moderator_setting = get_real_value(100, action[0][0])
                    wp1rpm_setting = get_real_value(2000, action[0][1])
                    frontend.getSliderWP1RPM().setValue(wp1rpm_setting)
                    print(f"MOD {frontend.getModPer()}")
                    print(moderator_setting)
                    if moderator_setting == 100 - frontend.getModPer():
                        frontend.fireChange()
                    frontend.getSliderRodPos().setValue(moderator_setting)
                    print(frontend.getPower())
                    # water_valve_setting = False if action[2] < 0 else True
                    # steam_valve_setting = False if action[3] < 0 else True
                    frontend.getBCLWV1().doClick() if int(action[0][2]) < 0 else frontend.getBOpWV1().doClick()
                    frontend.getBCLFV1().doClick() if int(action[0][3]) < 0 else frontend.getBOpFV1().doClick()
                    cprpm_setting = get_real_value(2000, action[0][4])
                    frontend.getSliderCPRPM().setValue(cprpm_setting)
                # TODO Adjust for different scenarios
                frontend.timeStep()
                # This is needed to get the same state for the environment. This is not the best way to do it.
                # Other option is to create a new env where only the frontend-calls are used, so we do not need
                # to do this here.
                obs, reward, done, info = vec_env.step(action)
                print(reward)
                if done:
                    print("Done")
                    break
            gateway.shutdown()
            p.kill()
            break
        except:
            pass


path = "models/scenario1/full_test_12_03/scenario1_ActionSpaceOption3Wrapper_None_NPPAutomationWrapper_RewardOption2Wrapper_TD3_full_test_12_03_2/best_model.zip"
scenario, alg, wrapper_maker = parse_information_from_path(path)
eval_frontend(scenario, path, alg, wrapper_maker)
