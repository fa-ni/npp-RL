from typing import List, Dict

from src.main.rl.wrapper.reward_calculations import calculate_reward_for_corridor


def prepare_critical_states_analysis(reactor_status_over_time: List[Dict]) -> List[Dict]:
    criticality_over_time = []
    critical_water = {"Reactor_WaterLevel": [1200, 2800, 1500, 2500], "Condenser_WaterLevel": [800, 7000, 1500, 6000]}
    critical_pressure = {"Reactor_Pressure": [450, 350], "Condenser_Pressure": [110, 80]}
    for item in reactor_status_over_time:
        criticality = {}
        for water_component in ["Reactor_WaterLevel", "Condenser_WaterLevel"]:
            if (
                item[water_component] <= critical_water[water_component][0]
                or item[water_component] > critical_water[water_component][1]
            ):
                criticality[water_component] = "Red"
            elif (
                item[water_component] <= critical_water[water_component][2]
                or item[water_component] > critical_water[water_component][3]
            ):
                criticality[water_component] = "Orange"
            else:
                criticality[water_component] = "Green"
        for pressure_component in ["Reactor_Pressure", "Condenser_Pressure"]:
            if item[pressure_component] > critical_pressure[pressure_component][0]:
                criticality[pressure_component] = "Red"
            elif item[pressure_component] >= critical_pressure[pressure_component][1]:
                criticality[pressure_component] = "Orange"
            else:
                criticality[pressure_component] = "Green"
        if item["Blow_Counter"] <= 3:
            criticality["Blow_Counter"] = "Red"
        elif item["Blow_Counter"] <= 5:
            criticality["Blow_Counter"] = "Orange"
        else:
            item["Blow_Counter"] = "Green"
        criticality_over_time.append(criticality)
    return criticality_over_time


def calculate_criticality_score_with_reward_functions(reactor_status_over_time: List[Dict]):
    """
    This function calculates the overall criticality score based on the status of different components over
    the whole evaluation time.
    """
    result = []
    for item in reactor_status_over_time:
        # 1200,2800 critical
        reactor_wl = calculate_reward_for_corridor(1200, 2800, 2100, item["Reactor_WaterLevel"])
        reactor_pressure = 1 if item["Reactor_Pressure"] < 350 else (0.5 if item["Reactor_Pressure"] < 450 else 0)

        # 800, 7000
        condenser_wl = calculate_reward_for_corridor(800, 5100, 2500, item["Condenser_WaterLevel"])
        condenser_pressure = 1 if item["Condenser_Pressure"] < 80 else (0.5 if item["Condenser_Pressure"] < 110 else 0)
        blow_counter = 1 if item["Blow_Counter"] > 10 else (0.5 if item["Blow_Counter"] > 5 else 0)
        result.append((reactor_wl + reactor_pressure + condenser_wl + condenser_pressure + blow_counter) / 5)
    return sum(result)


def calculate_score(criticality_of_states: List[Dict]) -> float:
    result = 0
    counts = {}
    list_of_all_states = []
    for item in criticality_of_states:
        list_of_all_states.extend(list(item.values()))
    for word in set(list_of_all_states):
        counts[word] = list_of_all_states.count(word)
    result = counts["Red"]
    result += counts["Orange"] * 0.5
    return result
