def calculate_score_for_corridor_with_perfect_number(lower: int, upper: int, perfect: int, number: float) -> float:
    if number < lower or number > upper:
        result = 0
    else:
        if number == perfect:
            result = 1
        elif number < perfect:
            result = (number - lower) / (perfect - lower)
        else:
            result = (upper - number) / (upper - perfect)
    return result


def calculate_roofed_reward(power_output: float) -> float:
    if power_output >= 700:
        result = 700 / power_output
    else:
        result = power_output / 700
    return result


import numpy as np


def calculate_score_for_corridor_without_perfect_number(
    lower_threshold: int, lower_bound: int, upper_threshold: int, upper_bound: int, value: float
) -> float:
    if value < lower_threshold:
        result = (value - lower_bound) / (lower_threshold - lower_bound)

    elif value >= upper_threshold:
        result = (upper_bound - value) / (upper_bound - upper_threshold)

    else:
        result = 1

    result = np.clip([result], 0, 1)
    return result[0]
