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
Starting State Option 2 was created with the following things in mind:
- Should be a "critical" state
- Waterlevel or pressure of the Rector is in the critical range
- Should be have a middle power output
- Was created with the help of the standard SOP policy
"""


# fmt: off
# Reactor Waterlevel critical low (red)
def create_starting_state_option2a():
    reactor = Reactor(
        water_level=1142.3,
        pressure=197.9,
        moderator_percent=75,
        overheated=False,
        melt_stage=1,
        poisoning_factor=deque(
            [81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 79, 79, 79, 79, 79, 79, 79,
             79, 79, 79, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 75,
             75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75],
            maxlen=100,
        ),
    )
    steam_valve1 = SteamValve(False, True)
    steam_valve2 = SteamValve(False, False)
    water_valve1 = WaterValve(False, True)
    water_valve2 = WaterValve(False, False)
    water_pump1 = Pump(rpm=694, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    condenser_pump = Pump(rpm=604, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    turbine = Turbine(False)
    condenser = Condenser(_waterLevel=3832.5, _pressure=58.3, _blown=False)
    generator = Generator(False, 349)
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

# Pressure Condenser critical high (red)
def create_starting_state_option2b():
    reactor = Reactor(
        water_level=1905.9,
        pressure=261.1,
        moderator_percent=71,
        overheated=False,
        melt_stage=1,
        poisoning_factor=deque(
            [77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 75, 75, 75, 75, 75, 75, 75,
             75, 75, 75, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
             73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 71,
             71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71],
            maxlen=100,
        ),
    )
    steam_valve1 = SteamValve(False, True)
    steam_valve2 = SteamValve(False, False)
    water_valve1 = WaterValve(False, True)
    water_valve2 = WaterValve(False, False)
    water_pump1 = Pump(rpm=910, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    condenser_pump = Pump(rpm=405, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    turbine = Turbine(False)
    condenser = Condenser(_waterLevel=2567.7, _pressure=120.5, _blown=False)
    generator = Generator(False, 351)
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

# Reactor Waterlevel high red
def create_starting_state_option2c():
    reactor = Reactor(
        water_level=2862.3,
        pressure=161.9,
        moderator_percent=75,
        overheated=False,
        melt_stage=1,
        poisoning_factor=deque(
            [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 79, 79, 79, 79, 79, 79, 79,
             79, 79, 79, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78,
             78, 78, 78, 78, 78, 78, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 75,
             75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75],
            maxlen=100,
        ),
    )
    steam_valve1 = SteamValve(False, True)
    steam_valve2 = SteamValve(False, False)
    water_valve1 = WaterValve(False, True)
    water_valve2 = WaterValve(False, False)
    water_pump1 = Pump(rpm=1000, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    condenser_pump = Pump(rpm=1270, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    turbine = Turbine(False)
    condenser = Condenser(_waterLevel=2432.8, _pressure=14.8, _blown=False)
    generator = Generator(False, 367)
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
