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
Starting State Option 3 was created with the following things in mind:
- Should be no critical state
- Should have a state that can not be produced with the GUI of the original simulation
"""


# fmt: off
def create_starting_state_option3():
    full_reactor = ReactorCreatorService().create_standard_full_reactor()
    full_reactor.reactor.water_level = 2800
    full_reactor.condenser.water_level = 4800
    return full_reactor
