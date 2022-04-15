# 1. CR/Moderator Percent 2. WP1 RPM 3. WV1 4. SV1 5. CP RPM
from src.main.rl.utils.utils import get_scaled_value


# 1. CR/Moderator Percent 2. WP1 RPM 3. WV1 4. SV1 5. CP RPM
def get_actions_sop(length: int) -> list:
    result = []
    if length == 249:
        result = [get_scaled_value(100, 0), get_scaled_value(2000, 0), get_scaled_value(1, 0),
                  get_scaled_value(1, 0), get_scaled_value(2000, 2000)]
    elif length == 248:
        result = [get_scaled_value(100, 0), get_scaled_value(2000, 0), get_scaled_value(1, 0),
                  get_scaled_value(1, 0), get_scaled_value(2000, 2000)]
    elif length == 247:
        result = [get_scaled_value(100, 0), get_scaled_value(2000, 0), get_scaled_value(1, 0),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1700)]
    # 1600 WP1 RPM reached
    elif length == 246:
        result = [get_scaled_value(100, 0), get_scaled_value(2000, 0), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    # WaterValve 1 opened
    elif length == 245:
        result = [get_scaled_value(100, 0), get_scaled_value(2000, 400), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    # 200 WP1 RPM reached
    elif length in [244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231]:
        result = [get_scaled_value(100, 0), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    elif length < 231 and length >= 223:
        factor = 231 - length
        result = [get_scaled_value(100, factor), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]

    elif length == 222:
        result = [get_scaled_value(100, 7), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    elif length == 221:
        result = [get_scaled_value(100, 6), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    elif length == 220:
        result = [get_scaled_value(100, 5), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    elif length == 219:
        result = [get_scaled_value(100, 5), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    elif length == 218:
        result = [get_scaled_value(100, 5), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 0), get_scaled_value(2000, 1600)]
    # Stabilized by 2100 mm waterlevel
    elif length == 217:
        result = [get_scaled_value(100, 5), get_scaled_value(2000, 200), get_scaled_value(1, 1),
                  get_scaled_value(1, 1), get_scaled_value(2000, 1600)]
    # Steam Valve 1 opened
    elif length < 217 and length >= 155:
        factor = 217 - length
        wp1_rpm = 400 + 35 * factor if 400 + 35 * factor < 1800 else 1800
        result = [get_scaled_value(100, 5 + factor), get_scaled_value(2000, wp1_rpm), get_scaled_value(1, 1),
                  get_scaled_value(1, 1), get_scaled_value(2000, 1600)]

    elif length < 155 and length >= 148:
        factor = 155 - length
        result = [get_scaled_value(100, 67 - factor), get_scaled_value(2000, 1900), get_scaled_value(1, 1),
                  get_scaled_value(1, 1), get_scaled_value(2000, 1600)]
        print("YYY")
    # 700 MW Power Output reached
    elif length < 148 and length >= 142:
        factor = 148 - length
        result = [get_scaled_value(100, 67 - factor), get_scaled_value(2000, 1780), get_scaled_value(1, 1),
                  get_scaled_value(1, 1), get_scaled_value(2000, 1600)]
    # print("YYY")
    # Stabilize around 2100 mm
    elif length < 142:
        result = [get_scaled_value(100, 60), get_scaled_value(2000, 1612), get_scaled_value(1, 1),
                  get_scaled_value(1, 1), get_scaled_value(2000, 1600)]
    else:
        print("Finished")
    return result
