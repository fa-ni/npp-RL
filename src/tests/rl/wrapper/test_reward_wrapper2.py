from src.main.rl.utils.utils import delete_env_id
from src.main.rl.wrapper.reward_wrapper2 import RewardOption2Wrapper
import mock
import gym


def set_up():
    # version = request.node.get_closest_marker("number").args[0]
    env_id = f"test-v1"  # {version}"
    delete_env_id(env_id)
    scenario = "src.main.rl.envs.scenario2:Scenario2"
    gym.register(id=env_id, entry_point=scenario)
    env = gym.make(env_id)
    env.reset()
    wrapper = RewardOption2Wrapper(env)
    return wrapper


@mock.patch("src.main.rl.wrapper.reward_wrapper2.calculate_roofed_reward")
@mock.patch("src.main.rl.wrapper.reward_wrapper2.is_done")
def test_reward_wrapper2(mock1, mock2):
    mock1.return_value = False
    mock2.return_value = 0.5
    wrapper = set_up()
    actual = wrapper.reward(0)
    expected = 0.5
    assert mock1.call_count == 1
    assert mock2.call_count == 1
    assert actual == expected
