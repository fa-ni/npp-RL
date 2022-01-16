from gym import Wrapper
from gym.spaces import MultiBinary


class ActionSpaceOption3Wrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 1. WP1 RPM 2. CR/Moderator Percent 3. WV1
        self.action_space = MultiBinary(5)
