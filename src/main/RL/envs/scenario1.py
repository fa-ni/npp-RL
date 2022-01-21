import numpy as np
from gym import Env
from gym.spaces import MultiBinary, Box

from src.main.services.BackgroundStepService import BackgroundStepService
from src.main.services.ReactorCreatorService import ReactorCreatorService


class Scenario1(Env):
    # Scenario 1 with Action Space Option 1 and Observation Space Option1

    def __init__(self):
        # 1. moderator_percent 2. WP1 RPM
        self.action_space = Box(np.array([-1, -1]).astype(np.float32), np.array([1, 1]).astype(np.float32))
        self.observation_space = Box(np.array([-1]).astype(np.float32), np.array([1]).astype(np.float32))
        self.length = 250

    def step(self, action):
        done = False
        reward = 0
        self.length -= 1
        self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
        self.state.full_reactor.steam_valve1.status = True
        self.state.full_reactor.water_valve1.status = True

        # (normalized value + 1)* (max_value/2)
        # Standard/Minimal Actions
        # Option 1 Actions
        moderator_percent_setting = (action[0] + 1) * (100 / 2)
        wp_rpm_setting = (action[1] + 1) * (2000 / 2)
        self.state.full_reactor.reactor.moderator_percent = (
            100 - self.state.full_reactor.reactor.moderator_percent + moderator_percent_setting
        )
        self.state.full_reactor.water_pump1.rpm_to_be_set = wp_rpm_setting
        # Necessary for Action Space Option 1
        if len(action) == 2:
            # This is necessary as you cannot override the state from this environment in any of the wrappers
            if self.length == 249:
                self.state.full_reactor.condenser_pump.rpm = 1600
                self.state.full_reactor.steam_valve1.status = True
                self.state.full_reactor.water_valve1.status = True
            self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
            self.state.full_reactor.steam_valve1.status = True
            self.state.full_reactor.water_valve1.status = True
        # Necessary for Action Space Option 2
        if len(action) == 3:  # TODO
            # This is necessary as Iyou cannot override the state from this environment in any of the wrappers
            if self.length == 249:
                self.state.full_reactor.condenser_pump.rpm = 1600
                self.state.full_reactor.steam_valve1.status = True
            self.state.full_reactor.condenser_pump.rpm_to_be_set = 1600
            self.state.full_reactor.steam_valve1.status = True
            water_valve_setting = False if action[2] < 0 else True
            self.state.full_reactor.water_valve1.status = water_valve_setting
        # Necessary for Action Space Option 3
        if len(action) == 5:
            water_valve_setting = False if action[2] < 0 else True
            steam_valve_setting = False if action[3] < 0 else True
            condenser_rpm_setting = (action[4] + 1) * (2000 / 2)
            self.state.full_reactor.water_valve1.status = water_valve_setting
            self.state.full_reactor.steam_valve1.status = steam_valve_setting
            self.state.full_reactor.condenser_pump.rpm_to_be_set = condenser_rpm_setting
        self.state.time_step(1)

        calc_reward = self.state.full_reactor.generator.power / 800
        reward += calc_reward  # TODO #calc_reward if calc_reward < 1 else 1
        if (
            self.state.full_reactor.reactor.overheated
            or self.state.full_reactor.reactor.is_blown()
            or self.state.full_reactor.generator.is_blown()
            or self.state.full_reactor.condenser.is_blown()
            or self.state.full_reactor.water_pump1.is_blown()
            or self.state.full_reactor.condenser_pump.is_blown()
            or self.length <= 0
            or self.state.full_reactor.water_pump1.rpm < 0
        ):
            done = True

        normalized_obs = 2 * (self.state.full_reactor.generator.power / 800) - 1
        return [
            # Might need to change if we dont want to have binary for first observation
            np.array([normalized_obs]),
            reward,
            done,
            {},
        ]

    def render(self):
        pass

    def reset(self):
        self.state = None
        self.state = BackgroundStepService(ReactorCreatorService.create_standard_full_reactor())
        self.moderator_percent = 100
        self.length = 250
        return np.array([float(-1)])
