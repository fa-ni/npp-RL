import numpy as np
from gym import Env
from gym.spaces import Box, MultiDiscrete

from src.main.dto.FullReactor import FullReactor
from src.main.rl.utils.reactor_starting_states import get_reactor_starting_state
from src.main.rl.utils.utils import is_done
from src.main.services.BackgroundStepService import BackgroundStepService
from src.main.services.ReactorCreatorService import ReactorCreatorService


class Scenario3(Env):
    # Scenario 3 with MultiDiscrete Action Spaces
    # if no wrapper is specified this will use ActionSpaceOption 1 and ObservationSpaceOption1
    def __init__(self, starting_state=None, length=250):
        # 1. moderator_percent 2. WP1 RPM
        self.action_space = MultiDiscrete([9, 9])
        self.observation_space = Box(np.array([-1]).astype(np.float32), np.array([1]).astype(np.float32))
        self.length = length
        # The key is the action value and the value is the mapping for the actual change for the real value.
        self.get_moderator_percentage_change = {0: -10, 1: -5, 2: -3, 3: -1, 4: 0, 5: 1, 6: 3, 7: 5, 8: 10}
        self.get_pump_change = {0: -200, 1: -100, 2: -50, 3: -25, 4: 0, 5: 25, 6: 50, 7: 100, 8: 200}
        self.starting_state = starting_state

    def step(self, action):
        reward = 0
        self.length -= 1

        # Standard/Minimal Actions (Option 1 Actions)
        moderator_percent_setting = self.get_moderator_percentage_change[action[0]]
        wp_rpm_setting = self.get_pump_change[action[1]]
        self.state.full_reactor.reactor.moderator_percent = (
            100 - self.state.full_reactor.reactor.moderator_percent + moderator_percent_setting
        )
        if wp_rpm_setting + self.state.full_reactor.water_pump1.rpm_to_be_set > 0:
            self.state.full_reactor.water_pump1.rpm_to_be_set += wp_rpm_setting
        else:
            self.state.full_reactor.water_pump1.rpm_to_be_set += 0

        # Necessary for Action Space Option 1
        if len(action) == 2:
            self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
            self.state.full_reactor.steam_valve1.status = True
            self.state.full_reactor.water_valve1.status = True
        # Necessary for Action Space Option 2
        if len(action) == 3:
            self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
            self.state.full_reactor.steam_valve1.status = True
            water_valve_setting = False if action[2] == 0 else True

            self.state.full_reactor.water_valve1.status = water_valve_setting
        # Necessary for Action Space Option 3
        if len(action) == 5:
            water_valve_setting = False if action[2] == 0 else True
            steam_valve_setting = False if action[3] == 0 else True
            condenser_rpm_setting = self.get_pump_change[action[4]]
            self.state.full_reactor.water_valve1.status = water_valve_setting
            self.state.full_reactor.steam_valve1.status = steam_valve_setting
            self.state.full_reactor.condenser_pump.rpm_to_be_set += condenser_rpm_setting
        self.state.time_step(1)

        done = is_done(self.state.full_reactor, self.length)
        if not done:
            calc_reward = self.state.full_reactor.generator.power / 700
            reward += calc_reward

        info = {
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
            done,
            info,
        ]

    def render(self):
        pass

    def reset(self):
        self.state = None
        self.length = self.length
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
