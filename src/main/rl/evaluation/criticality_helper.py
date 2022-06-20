from typing import List, Dict

from src.main.rl.wrapper.reward_calculations import (
    calculate_score_for_corridor_with_perfect_number,
    calculate_score_for_corridor_without_perfect_number,
)


def calculate_criticality_score_with_reward_functions(reactor_status_over_time: List[Dict]) -> (float, float):
    """
    This function calculates the overall criticality score based on the status of different components over
    the whole evaluation time.
    """
    reactor_wl_score = []
    reactor_pressure_score = []

    codenser_wl_score = []
    condenser_pressure_score = []
    blow_counter_score = []

    for item in reactor_status_over_time:
        reactor_wl_score.append(
            calculate_score_for_corridor_with_perfect_number(1000, 2900, 2100, item["Reactor_WaterLevel"])
        )
        reactor_pressure_score.append(
            1 if item["Reactor_Pressure"] < 350 else (500 - item["Reactor_Pressure"]) / (500 - 350)
        )
        codenser_wl_score.append(
            calculate_score_for_corridor_without_perfect_number(1500, 300, 4400, 5300, item["Condenser_WaterLevel"])
        )
        condenser_pressure_score.append(
            1 if item["Condenser_Pressure"] < 80 else (140 - item["Condenser_Pressure"]) / (140 - 80)
        )
        blow_counter_score.append(1 if item["Blow_Counter"] > 10 else item["Blow_Counter"] / 10)

    single_minimum_of_all = min(
        reactor_wl_score + reactor_pressure_score + codenser_wl_score + condenser_pressure_score + blow_counter_score
    )
    result_of_minimum_per_timestep = []
    for idx, item in enumerate(reactor_wl_score):
        result_of_minimum_per_timestep.append(
            min(
                item,
                reactor_pressure_score[idx],
                codenser_wl_score[idx],
                condenser_pressure_score[idx],
                blow_counter_score[idx],
            )
        )
    return single_minimum_of_all, sum(result_of_minimum_per_timestep)
