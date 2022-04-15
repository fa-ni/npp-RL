from collections import deque

from src.main.dto.Condenser import Condenser
from src.main.dto.FullReactor import FullReactor
from src.main.dto.Generator import Generator
from src.main.dto.Pump import Pump
from src.main.dto.Reactor import Reactor
from src.main.dto.SteamValve import SteamValve
from src.main.dto.Turbine import Turbine
from src.main.dto.WaterValve import WaterValve

"""
Starting State Option 1 was created with the following things in mind:
- Should be a "normal" state - no critical state
- Should be have a middle power output
- Was created with the help of the standard SOP policy
- Which was stopped after the power output reached 350 (or higher) and this state is now used
"""


def create_starting_state_option1():
    reactor = Reactor(
        water_level=2037.9,
        pressure=150.0,
        moderator_percent=66,
        overheated=False,
        melt_stage=1,
        poisoning_factor=deque(
            [
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                74,
                74,
                74,
                74,
                74,
                74,
                74,
                74,
                74,
                74,
                73,
                73,
                73,
                73,
                73,
                73,
                73,
                73,
                73,
                73,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                72,
                70,
                70,
                70,
                70,
                70,
                70,
                70,
                70,
                70,
                70,
                69,
                69,
                69,
                69,
                69,
                69,
                69,
                69,
                69,
                69,
                68,
                68,
                68,
                68,
                68,
                68,
                68,
                68,
                68,
                68,
                67,
                67,
                67,
                67,
                67,
                67,
                67,
                67,
                67,
                67,
                66,
                66,
                66,
                66,
                66,
                66,
                66,
                66,
                66,
                66,
            ],
            maxlen=100,
        ),
    )
    steam_valve1 = SteamValve(False, True)
    steam_valve2 = SteamValve(False, False)
    water_valve1 = WaterValve(False, True)
    water_valve2 = WaterValve(False, False)
    water_pump1 = Pump(rpm=1380, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    condenser_pump = Pump(rpm=1600, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    turbine = Turbine(False)
    condenser = Condenser(_waterLevel=3334.2, _pressure=7.8, _blown=False)
    generator = Generator(False, 355)
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
