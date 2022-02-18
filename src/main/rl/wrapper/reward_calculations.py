def calculate_reward_for_corridor(lower: int, upper: int, perfect: int, number: int) -> float:
    if number < lower or number > upper:
        result = 0
    else:
        result = 1
    # else:
    #    if number == perfect:
    #        result = 1
    #    elif number < perfect:
    #        result = (number - lower) / (perfect - lower)
    #    else:
    #        result = (number - perfect) / (upper - perfect)
    return result


def calculate_roofed_reward(power_output: int) -> float:
    if power_output >= 700:
        result = 700 / power_output
    else:
        result = power_output / 700
    return result
