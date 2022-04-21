import numpy as np
from gym import Wrapper

from src.main.rl.utils.utils import is_done
from src.main.services.NPPAutomationStepService import NPPAutomationStepService


class NPPAutomationWrapper(Wrapper):
    def step(self, action):
        self.env.step(action)
        reward = 0
        self.npp_automation.run()
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

    def reset(self):
        self.env.reset()
        self.npp_automation = NPPAutomationStepService(self.state)
        self.done = False
        return np.array([float(-1)])
