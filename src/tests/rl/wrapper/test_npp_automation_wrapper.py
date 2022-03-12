from unittest import mock

import numpy as np
from gym import register, make

from src.main.rl.utils.utils import delete_env_id
from src.main.rl.wrapper.npp_automation_wrapper import NPPAutomationWrapper
from src.main.services.NPPAutomationStepService import NPPAutomationStepService


def set_up():
    env_id = f"test-v1"
    delete_env_id(env_id)
    scenario = "src.main.rl.envs.scenario2:Scenario2"
    register(id=env_id, entry_point=scenario)
    env = make(env_id)
    wrapper = NPPAutomationWrapper(env)
    return wrapper


@mock.patch("src.main.rl.wrapper.npp_automation_wrapper.is_done")
def test_step(mock_is_done):
    mock_is_done.retun_value = False
    wrapper = set_up()
    wrapper.reset()
    with mock.patch.object(wrapper, "npp_automation", wraps=wrapper.npp_automation) as npp_automation:
        with mock.patch.object(wrapper, "env", wraps=wrapper.env) as env:
            wrapper.step(action=[0, 1])
            wrapper.step(action=[1, 1])
            wrapper.step(action=[1, 1])
            assert mock_is_done.call_count == 3
            assert npp_automation.run.call_count == 3
            assert env.step.call_count == 3


def test_reset():
    wrapper = set_up()
    actual = wrapper.reset()
    assert actual == np.array([-1])
    assert wrapper.npp_automation != None
    assert isinstance(wrapper.npp_automation, NPPAutomationStepService)
