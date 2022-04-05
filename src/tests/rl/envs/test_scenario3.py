import numpy as np
from gym.spaces import Box, MultiDiscrete

from src.main.rl.envs.scenario3 import Scenario3


def test_scenario3_reset():
    scenario = Scenario3()
    actual = scenario.reset()
    assert actual == np.array([-1])


def test_scenario3_init():
    scenario = Scenario3()
    assert scenario.action_space == MultiDiscrete([9, 9])
    assert scenario.observation_space == Box(np.array([-1]).astype(np.float32), np.array([1]).astype(np.float32))


def test_scenario3_step_with_two_actions():
    scenario = Scenario3()
    scenario.reset()
    step_one_result = scenario.step([5, 8])
    assert scenario.length == 249
    assert scenario.state.full_reactor.condenser_pump.rpm == 1600
    assert scenario.state.full_reactor.steam_valve1.status == True
    assert scenario.state.full_reactor.water_valve1.status == True
    assert scenario.state.full_reactor.condenser_pump.rpm_to_be_set == 1600
    assert scenario.state.full_reactor.reactor.moderator_percent == 99
    assert scenario.state.full_reactor.water_pump1.rpm_to_be_set == 200
    assert step_one_result == [np.array([-1.0]), 0.0, False, {}]
    step_two_result = scenario.step([6, 3])
    assert scenario.length == 248
    assert scenario.state.full_reactor.condenser_pump.rpm == 1600
    assert scenario.state.full_reactor.steam_valve1.status == True
    assert scenario.state.full_reactor.water_valve1.status == True
    assert scenario.state.full_reactor.condenser_pump.rpm_to_be_set == 1600
    assert scenario.state.full_reactor.reactor.moderator_percent == 96
    assert scenario.state.full_reactor.water_pump1.rpm_to_be_set == 175
    assert step_two_result == [np.array([-0.99]), 0.005714285714285714, False, {}]


def test_scenario3_step_with_three_actions():
    scenario = Scenario3()
    scenario.reset()
    step_one_result = scenario.step([4, 4, 1])
    assert scenario.length == 249
    assert scenario.state.full_reactor.condenser_pump.rpm == 1600
    assert scenario.state.full_reactor.steam_valve1.status == True
    assert scenario.state.full_reactor.water_valve1.status == True
    assert scenario.state.full_reactor.condenser_pump.rpm_to_be_set == 1600
    assert scenario.state.full_reactor.reactor.moderator_percent == 100
    assert scenario.state.full_reactor.water_pump1.rpm_to_be_set == 0
    assert step_one_result == [np.array([-1.0]), 0.0, False, {}]
    step_two_result = scenario.step([8, 8, 1])
    assert scenario.length == 248
    assert scenario.state.full_reactor.condenser_pump.rpm == 1600
    assert scenario.state.full_reactor.steam_valve1.status == True
    assert scenario.state.full_reactor.water_valve1.status == True
    assert scenario.state.full_reactor.condenser_pump.rpm_to_be_set == 1600
    assert scenario.state.full_reactor.reactor.moderator_percent == 90
    assert scenario.state.full_reactor.water_pump1.rpm_to_be_set == 200
    assert step_two_result == [np.array([-0.955]), 0.025714285714285714, False, {}]


def test_scenario3_step_with_five_actions():
    scenario = Scenario3()
    scenario.reset()
    step_one_result = scenario.step([2, 1, 0, 0, 4])
    assert scenario.length == 249
    assert scenario.state.full_reactor.steam_valve1.status == False
    assert scenario.state.full_reactor.water_valve1.status == False
    assert scenario.state.full_reactor.condenser_pump.rpm_to_be_set == 0
    assert scenario.state.full_reactor.reactor.moderator_percent == 100
    assert scenario.state.full_reactor.water_pump1.rpm_to_be_set == 0
    assert step_one_result == [np.array([-1.0]), 0.0, False, {}]
    step_two_result = scenario.step([0, 4, 1, 1, 6])
    assert scenario.length == 248
    assert scenario.state.full_reactor.steam_valve1.status == True
    assert scenario.state.full_reactor.water_valve1.status == True
    assert scenario.state.full_reactor.condenser_pump.rpm_to_be_set == 50
    assert scenario.state.full_reactor.reactor.moderator_percent == 100
    assert scenario.state.full_reactor.water_pump1.rpm_to_be_set == 0
    assert step_two_result == [np.array([-1]), 0.0, False, {}]
