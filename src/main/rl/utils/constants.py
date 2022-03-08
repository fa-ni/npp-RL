from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from src.main.rl.wrapper.action_wrapper2 import ActionSpaceOption2Wrapper
from src.main.rl.wrapper.action_wrapper3 import ActionSpaceOption3Wrapper
from src.main.rl.wrapper.obs_wrapper2 import ObservationOption2Wrapper
from src.main.rl.wrapper.obs_wrapper3 import ObservationOption3Wrapper
from src.main.rl.wrapper.obs_wrapper4 import ObservationOption4Wrapper
from src.main.rl.wrapper.obs_wrapper5 import ObservationOption5Wrapper

ALL_SCENARIOS = [
    "src.main.rl.envs.scenario1:Scenario1",
    "src.main.rl.envs.scenario2:Scenario2",
    "src.main.rl.envs.scenario3:Scenario3",
]
ALL_ACTION_WRAPPERS = [ActionSpaceOption2Wrapper, ActionSpaceOption3Wrapper]
ALL_OBSERVATION_WRAPPERS = [
    ObservationOption2Wrapper,
    ObservationOption3Wrapper,
    ObservationOption4Wrapper,
    ObservationOption5Wrapper,
]

# Action based information
scaling_factors_scenario_1 = [100, 2000, 1, 1, 2000]
scaling_factors_scenario2 = [1, 25, 1, 1, 25]
action_dimensions = ["Moderator Percent", "WP1 RPM", "WV1", "SV1", "CP RPM"]
# Observation based information
obs_scaling_factors = {
    1: [800],
    3: [800, 2000, 100],
    7: [800, 2000, 100, 2000, 1, 30],
    6: [800, 4000, 550, 8000, 180, 30],
    11: [800, 4000, 550, 8000, 180, 2000, 100, 2000, 1, 1, 30],
}
obs_dimensions = {
    1: ["Power Output"],
    3: ["Power Output", "WP1 RPM", "Moderator Percentage"],
    7: ["Power Output", "WP1 RPM", "Moderator Percentage", "CP RPM", "WV1", "SV1", "Blow Counter"],
    6: [
        "Power Output",
        "Reactor WaterLevel",
        "Reactor Pressure",
        "Condenser WaterLevel",
        "Condenser Pressure",
        "Blow Counter",
    ],
    11: [
        "Power Output",
        "Reactor WaterLevel",
        "Reactor Pressure",
        "Condenser WaterLevel",
        "Condenser Pressure",
        "WP1 RPM",
        "Moderator Percentage",
        "CP RPM",
        "WV1",
        "SV1",
        "Blow Counter",
    ],
}
alg_mapping = {"A2C": A2C, "DDPG": DDPG, "PPO": PPO, "SAC": SAC, "TD3": TD3}
