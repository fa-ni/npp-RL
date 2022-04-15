from collections import deque

from src.main.dto.Condenser import Condenser
from src.main.dto.FullReactor import FullReactor
from src.main.dto.Generator import Generator
from src.main.dto.Pump import Pump
from src.main.dto.Reactor import Reactor
from src.main.dto.SteamValve import SteamValve
from src.main.dto.Turbine import Turbine
from src.main.dto.WaterValve import WaterValve
from src.main.services.ReactorCreatorService import ReactorCreatorService

"""
Starting State Option 1 was created with the following things in mind:
- Should be a "normal" state - no critical state
- Should be have a middle power output
- Was created with the help of the standard SOP policy
- Which was stopped after the power output reached 350 (or higher) and this state is now used
"""


# fmt: off
def create_starting_state_option3():
    reactor = Reactor(
        water_level=2100.3,
        pressure=80.3,
        moderator_percent=77,
        overheated=False,
        melt_stage=1,
        poisoning_factor=deque(
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            maxlen=100,
        ),
    )
    steam_valve1 = SteamValve(False, True)
    steam_valve2 = SteamValve(False, False)
    water_valve1 = WaterValve(False, True)
    water_valve2 = WaterValve(False, False)
    water_pump1 = Pump(rpm=800, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    condenser_pump = Pump(rpm=500, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    turbine = Turbine(False)
    condenser = Condenser(_waterLevel=5000.7, _pressure=15.8, _blown=False)
    generator = Generator(False, 350)
    full_reactor = FullReactor(
        reactor=reactor,
        steam_valve1=steam_valve1,
        steam_valve2=steam_valve2,
        water_valve1=water_valve1,
        water_valve2=water_valve2,
        water_pump1=water_pump1,
        water_pump2=water_pump2,
        condenser_pump=condenser_pump,
        turbine=turbine,
        condenser=condenser,
        generator=generator,
    )
    return full_reactor


def create_starting_state_option3a():
    full_reactor = ReactorCreatorService().create_standard_full_reactor()
    full_reactor.reactor.water_level = 2500
    full_reactor.condenser.water_level = 4500
    return full_reactor


def create_starting_state_option3b():
    full_reactor = ReactorCreatorService().create_standard_full_reactor()
    full_reactor.reactor.water_level = 2800
    full_reactor.condenser.water_level = 4800
    return full_reactor
