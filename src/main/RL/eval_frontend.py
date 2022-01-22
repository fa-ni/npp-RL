from py4j.java_gateway import JavaGateway
from stable_baselines3 import PPO


def eval_frontend(env):
    gateway = JavaGateway()  # connect to the JVM
    entry = gateway.entry_point
    frontend = entry.getNPPUI()

    model = PPO.load("./best_model/best_model.zip")
    score = []
    obs = env.reset()
    reward_total = 0
    for i in range(300):
        action, _states = model.predict(obs)
        moderator_setting = -1 if action[0][0] == 0 else 1

        wp1rpm_setting = -25 if action[0][1] == 0 else +25
        frontend.getSliderWP1RPM().setValue(frontend.getSliderWP1RPM().getValue() + wp1rpm_setting)
        frontend.getSliderRodPos().setValue(frontend.getSliderRodPos().getValue() + moderator_setting)
        frontend.getBCLWV1().doClick() if int(action[0][2]) == 0 else frontend.getBOpWV1().doClick()
        frontend.getBCLFV1().doClick() if int(action[0][3]) == 0 else frontend.getBOpFV1().doClick()
        cprpm_setting = -25 if action[0][4] == 0 else +25
        frontend.getSliderCPRPM().setValue(frontend.getSliderCPRPM().getValue() + cprpm_setting)
        frontend.timeStep()
        score.append(int(frontend.getPower()))
        # This is needed to get the same state for the environment. This is not the best way to do it.
        # Other option is to create a new env where only the fronend-calls are used, so we do not need
        # to do this here.
        obs, reward, done, info = env.step(action)
        reward_total += reward
        if done:
            print("Done")
            break
