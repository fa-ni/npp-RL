import numpy as np
from gym import Wrapper
from gym.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class ObservationOption3Wrapper(Wrapper):
    # def step_wait(self) -> VecEnvStepReturn:
    #    pass

    def __init__(self, env):
        super().__init__(env)
        # 1. Power Output 2. WP1 RPM 3. CP/moderator Percentage
        self.observation_space = Box(
            np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32), np.array([1, 1, 1, 1, 1, 1]).astype(np.float32)
        )

    def step(self, action):
        original_result = self.unwrapped.step(action)
        normalized_wp1_rpm = 2 * (self.state.full_reactor.water_pump1.rpm / 2000) - 1
        normalized_moderator_percent = (
            2 * (self.state.full_reactor.reactor.moderator_percent / 100) - 1
        )  # TODO Check ob das so passt, oder halt nicht, da hier mit dem minus
        normalized_cp_rpm = 2 * (self.state.full_reactor.condenser_pump.rpm / 2000) - 1
        normalized_wv1_status = 2 * (int(self.state.full_reactor.water_valve1.status)) - 1
        normalized_sv1_status = 2 * (int(self.state.full_reactor.steam_valve1.status)) - 1

        original_result[0] = np.append(
            original_result[0],
            np.array(
                [
                    float(normalized_wp1_rpm),
                    float(normalized_moderator_percent),
                    float(normalized_cp_rpm),
                    float(normalized_wv1_status),
                    float(normalized_sv1_status),
                ]
            ),
        )
        return tuple(original_result)

    def reset(self):
        sv1_status = -1
        cp_rpm_status = -1
        wv1_status = -1
        # TODO Check mit anderem Action Space
        if self.action_space.shape[0] <= 3:
            cp_rpm_status = 2 * (1600 / 2000) - 1
            sv1_status = 1
        if self.action_space.shape[0] == 2:
            wv1_status = 1
        return np.append(
            self.unwrapped.reset(),
            np.array([float(-1), float(1), float(cp_rpm_status), float(wv1_status), float(sv1_status)]),
        )
