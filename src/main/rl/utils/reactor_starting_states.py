from src.main.dto.FullReactor import FullReactor
from src.main.dto.Condenser import Condenser
from src.main.dto.Generator import Generator
from src.main.dto.Pump import Pump
from src.main.dto.SteamValve import SteamValve
from src.main.dto.Turbine import Turbine
from src.main.dto.WaterValve import WaterValve
from src.main.dto.Reactor import Reactor


def get_reactor_starting_state(starting_state):
    reactor = Reactor(
        water_level=1848.523036335798, pressure=34.80510158316655, moderator_percent=85, overheated=False, melt_stage=1
    )
    steam_valve1 = SteamValve(False, True)
    steam_valve2 = SteamValve(False, False)
    water_valve1 = WaterValve(False, False)
    water_valve2 = WaterValve(False, False)
    water_pump1 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    condenser_pump = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    turbine = Turbine(False)
    condenser = Condenser(_waterLevel=3997.3749996495626, _pressure=3.7203894204932704, _blown=False)
    generator = Generator(False, 77)
    water_pump1.rpm = 321
    condenser_pump.rpm = 351
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
    # reactor = Reactor(water_level=2500.523036335798, pressure=54.80510158316655, moderator_percent=90, overheated=False,
    #                  melt_stage=1)
    # steam_valve1 = SteamValve(False, False)
    # steam_valve2 = SteamValve(False, False)
    # water_valve1 = WaterValve(False, True)
    # water_valve2 = WaterValve(False, False)
    # water_pump1 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    # water_pump2 = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    # condenser_pump = Pump(rpm=0, max_rpm=2000, upper_rpm_threshold=1800, blown=False)
    # turbine = Turbine(False)
    # condenser = Condenser(_waterLevel=3000.3749996495626, _pressure=3.7203894204932704, _blown=False)
    # generator = Generator(False, 77)
    # water_pump1.rpm = 321
    # condenser_pump.rpm = 351
    # full_reactor = FullReactor(
    #    reactor=reactor,
    #    steam_valve1=steam_valve1,
    #    steam_valve2=steam_valve2,
    #    water_valve1=water_valve1,
    #    water_valve2=water_valve2,
    #    water_pump1=water_pump1,
    #    water_pump2=water_pump2,
    #    condenser_pump=condenser_pump,
    #    turbine=turbine,
    #    condenser=condenser,
    #    generator=generator,
    # )
    # return full_reactor
