from src.main.RL.wrapper.ActionSpaceOption2Wrapper import ActionSpaceOption2Wrapper
from src.main.RL.wrapper.ActionSpaceOption3Wrapper import ActionSpaceOption3Wrapper
from src.main.RL.wrapper.ObservationOption2Wrapper import ObservationOption2Wrapper
from src.main.RL.wrapper.ObservationOption3Wrapper import ObservationOption3Wrapper
from src.main.RL.wrapper.ObservationOption4Wrapper import ObservationOption4Wrapper
from src.main.RL.wrapper.ObservationOption5Wrapper import ObservationOption5Wrapper

ALL_SCENARIOS = [
    "src.main.RL.envs.scenario1:Scenario1",
    "src.main.RL.envs.scenario2:Scenario2",
    "src.main.RL.envs.scenario3:Scenario3",
]
ALL_ACTION_WRAPPERS = [ActionSpaceOption2Wrapper, ActionSpaceOption3Wrapper]
ALL_OBSERVATION_WRAPPERS = [
    ObservationOption2Wrapper,
    ObservationOption3Wrapper,
    ObservationOption4Wrapper,
    ObservationOption5Wrapper,
]
