from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from src.main.rl.evaluation.starting_state_option1 import create_starting_state_option1
from src.main.rl.evaluation.starting_state_option2 import (
    create_starting_state_option2c,
    create_starting_state_option2,
    create_starting_state_option2a,
    create_starting_state_option2b,
)
from src.main.rl.evaluation.starting_state_option3 import (
    create_starting_state_option3,
    create_starting_state_option3a,
    create_starting_state_option3b,
)
from src.main.rl.evaluation.wrapper.noise.delay_wrapper import (
    DelayNoiseWrapperOption2,
    DelayNoiseWrapperOption1,
)
from src.main.rl.evaluation.wrapper.noise.obs_varies_wrapper import (
    ObservationVariesNoiseWrapper1,
    ObservationVariesPositiveNoiseWrapper,
    ObservationVariesNegativeNoiseWrapper,
)
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
ALL_OBS_NOISE_WRAPPERS = [
    ObservationVariesPositiveNoiseWrapper,
    ObservationVariesNegativeNoiseWrapper,
    ObservationVariesNoiseWrapper1,
]
ALL_DELAY_NOISE_WRAPPERS = [DelayNoiseWrapperOption1, DelayNoiseWrapperOption2]

STARTING_STATE_OPTION1 = [create_starting_state_option1]
STARTING_STATE_OPTION2 = [
    create_starting_state_option2,
    create_starting_state_option2a,
    create_starting_state_option2b,
    create_starting_state_option2c,
]
STARTING_STATE_OPTION3 = [create_starting_state_option3, create_starting_state_option3a, create_starting_state_option3b]

# Action based information
scaling_factors_scenario_1 = [100, 2000, 1, 1, 2000]
scaling_factors_scenario2 = [1, 25, 1, 1, 25]
moderator_percentage_change_scenario3 = {0: -10, 1: -5, 2: -3, 3: -1, 4: 0, 5: 1, 6: 3, 7: 5, 8: 10}
pump_change_scenario3 = {0: -200, 1: -100, 2: -50, 3: -25, 4: 0, 5: 25, 6: 50, 7: 100, 8: 200}
scaling_factors_scenario3 = [
    moderator_percentage_change_scenario3,
    pump_change_scenario3,
    {1: 1, 0: 0},
    {1: 1, 0: 0},
    pump_change_scenario3,
]
action_dimensions = ["Moderator Percent", "WP1 RPM", "WV1", "SV1", "CP RPM"]
action_dimensions_german = ["Regelstäbe", "Wasserpumpe 1 RPM", "Wasserventil 1", "Dampfventil 1", "Kondensator RPM"]
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

obs_dimensions_german = {
    6: [
        "Leistungsabgabe",
        "Wasserlevel Reaktor",
        "Druck Reaktor",
        "Wasserlevel Kondensator",
        "Druck Kondensator",
        "Blow Counter",
    ],
}
alg_mapping = {"A2C": A2C, "DDPG": DDPG, "PPO": PPO, "SAC": SAC, "TD3": TD3}

color_mapping = {
    "standard": "#1D2D5F",
    "red": "#F65E5D",
    "blue": "#1D2D5F",
    "scenario1": "#1D2D5F",
    "scenario2": "#F65E5D",
    "scenario3": "#FFBC47",
    "None": "#1D2D5F",
    "ActionSpaceOption1Wrapper": "#1D2D5F",
    "NPPAutomationWrapper": "#1D2D5F",
    "ActionSpaceOption2Wrapper": "#F65E5D",
    "ActionSpaceOption3Wrapper": "#FFBC47",
}
