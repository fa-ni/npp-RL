import numpy as np
from gym import Wrapper
from gym.spaces import Box

from src.main.dto.FullReactor import FullReactor


class ObservationOption4Wrapper(Wrapper):
    """
    Wrapper used to have 3 dimensions in the Observation Space.
    1. Power Output 2. Reactor WaterLevel 3. Reactor Pressure 4. Condenser WaterLevel 5. Condenser Pressure 6. Blow Counter
    """

    def __init__(self, env):
        super().__init__(env)
        # 1. Power Output 2. Reactor WaterLevel 3. Reactor Pressure 4. Condenser WaterLevel 5. Condenser Pressure 6. Blow Counter
        self.observation_space = Box(
            np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32), np.array([1, 1, 1, 1, 1, 1]).astype(np.float32)
        )

    def step(self, action):
        original_result = list(self.env.step(action))
        normalized_reactor_water_level = 2 * (self.state.full_reactor.reactor.water_level / 4000) - 1
        normalized_reactor_pressure = 2 * (self.state.full_reactor.reactor.pressure / 550) - 1
        normalized_condenser_water_level = 2 * (self.state.full_reactor.condenser.water_level / 8000) - 1
        normalized_condenser_pressure = 2 * (self.state.full_reactor.condenser.pressure / 180) - 1
        normalized_blow_counter = 2 * (self.state.full_reactor.water_pump1.blow_counter / 30) - 1
        original_result[0] = np.append(
            original_result[0],
            np.array(
                [
                    float(normalized_reactor_water_level),
                    float(normalized_reactor_pressure),
                    float(normalized_condenser_water_level),
                    float(normalized_condenser_pressure),
                    float(normalized_blow_counter),
                ]
            ),
        )
        return tuple(original_result)

    def reset(self):
        self.env.reset()
        return_values = get_return_values_for_starting_state(self.state.full_reactor)
        return return_values


def get_return_values_for_starting_state(full_reactor: FullReactor):
    normalized_power = 2 * (full_reactor.generator.power / 800) - 1
    normalized_reactor_water_level = 2 * (full_reactor.reactor.water_level / 4000) - 1
    normalized_reactor_pressure = 2 * (full_reactor.reactor.pressure / 550) - 1
    normalized_condenser_water_level = 2 * (full_reactor.condenser.water_level / 8000) - 1
    normalized_condenser_pressure = 2 * (full_reactor.condenser.pressure / 180) - 1
    normalized_blow_counter = 2 * (full_reactor.water_pump1.blow_counter / 30) - 1
    return np.array(
        [
            float(normalized_power),
            float(normalized_reactor_water_level),
            float(normalized_reactor_pressure),
            float(normalized_condenser_water_level),
            float(normalized_condenser_pressure),
            float(normalized_blow_counter),
        ]
    )
