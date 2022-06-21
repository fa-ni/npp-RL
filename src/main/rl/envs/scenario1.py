import numpy as np
from gym import Env
from gym.spaces import Box

from src.main.dto.FullReactor import FullReactor
from src.main.rl.utils.utils import get_real_value, is_done
from src.main.services.BackgroundStepService import BackgroundStepService
from src.main.services.ReactorCreatorService import ReactorCreatorService


class Scenario1(Env):
    """
    This openAI gym environment is the first scenario.
    If no wrapper is specified this will use ActionSpaceOption 1 and ObservationSpaceOption1.
    It uses a box action space. So it will accept continuous action in the interval [-1,1].
    The default length after a episode is over is 250. This can be overwritten with the length param.
    The starting state is the npp simulation with everything set to zero (with ActionSpaceOption3Wrapper). A different starting state can be overhanded with the XX param. If there is no ActionSpace Wrapper used or ActionSpaceOption2Wrapper then some default values are set before the start of the first timestep.
    """

    def __init__(self, starting_state=None, length=250):
        # 1. moderator_percent 2. WP1 RPM
        self.action_space = Box(np.array([-1, -1]).astype(np.float32), np.array([1, 1]).astype(np.float32))
        self.observation_space = Box(np.array([-1]).astype(np.float32), np.array([1]).astype(np.float32))
        self.done = False
        self.length_initial = length
        self.length = length

        self.starting_state = starting_state

    def step(self, action):
        reward = 0
        self.length -= 1
        # Standard/Minimal Actions
        moderator_percent_setting = get_real_value(100, action[0])
        wp_rpm_setting = get_real_value(2000, action[1])
        self.state.full_reactor.reactor.moderator_percent = moderator_percent_setting
        self.state.full_reactor.water_pump1.rpm_to_be_set = wp_rpm_setting
        # Necessary for Action Space Option 1
        if len(action) == 2:
            self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
            self.state.full_reactor.steam_valve1.status = True
            self.state.full_reactor.water_valve1.status = True
        # Necessary for Action Space Option 2
        if len(action) == 3:
            self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
            self.state.full_reactor.steam_valve1.status = True
            water_valve_setting = False if action[2] < 0 else True
            self.state.full_reactor.water_valve1.status = water_valve_setting
        # Necessary for Action Space Option 3
        if len(action) == 5:
            water_valve_setting = False if action[2] < 0 else True
            steam_valve_setting = False if action[3] < 0 else True
            condenser_rpm_setting = get_real_value(2000, action[4])
            self.state.full_reactor.water_valve1.status = water_valve_setting
            self.state.full_reactor.steam_valve1.status = steam_valve_setting
            self.state.full_reactor.condenser_pump.rpm_to_be_set = condenser_rpm_setting
        self.state.time_step(1)

        self.done = is_done(self.state.full_reactor, self.length)
        if not self.done:
            calc_reward = self.state.full_reactor.generator.power / 700
            reward += calc_reward

        info = {
            "Power_Output": self.state.full_reactor.generator.power,
            "Reactor_WaterLevel": self.state.full_reactor.reactor.water_level,
            "Reactor_Pressure": self.state.full_reactor.reactor.pressure,
            "Condenser_WaterLevel": self.state.full_reactor.condenser.water_level,
            "Condenser_Pressure": self.state.full_reactor.condenser.pressure,
            "Blow_Counter": self.state.full_reactor.water_pump1.blow_counter,
            "Condensator_Pump Blown": self.state.full_reactor.condenser_pump.is_blown(),
            "Water_Pump Blown": self.state.full_reactor.water_pump1.is_blown(),
        }
        normalized_obs = 2 * (self.state.full_reactor.generator.power / 800) - 1
        return [
            # Might need to change if we dont want to have binary for first observation
            np.array([normalized_obs]),
            reward,
            self.done,
            info,
        ]

        def render(self, **kwargs):
            pass

    def reset(self):
        self.state = None
        self.done = False
        self.length = self.length_initial
        if self.starting_state:
            self.state = BackgroundStepService(self.starting_state)
        else:
            self.state = BackgroundStepService(ReactorCreatorService.create_standard_full_reactor())
            # For ActionSpaceOption 1 we need to set these values in the beginning.
            # If we have a different ActionSpaceOption we will override the values again in the
            # action_wrapper.
            self.state.full_reactor.condenser_pump.rpm = 1600
            self.state.full_reactor.steam_valve1.status = True
            self.state.full_reactor.water_valve1.status = True
        return_values = get_return_values_for_starting_state(self.state.full_reactor)
        return return_values


def get_return_values_for_starting_state(full_reactor: FullReactor):
    normalized_power = 2 * (full_reactor.generator.power / 800) - 1
    return np.array([float(normalized_power)])
