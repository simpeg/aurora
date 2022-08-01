"""
2022-07-31
prototype module to enable handling of channel names that depart from the usual
hexy convention.

"""

DEFAULT_CHANNEL_MAP = {
    "hx": "hx",
    "hy": "hy",
    "hz": "hz",
    "ex": "ex",
    "ey": "ey",
}

LEMI_CHANNEL_MAP_12 = {
    "hx": "bx",
    "hy": "by",
    "hz": "bz",
    "ex": "e1",
    "ey": "e2",
}

LEMI_CHANNEL_MAP_34 = {
    "hx": "bx",
    "hy": "by",
    "hz": "bz",
    "ex": "e3",
    "ey": "e4",
}

THE_BEATLES = {
    "hx": "john",
    "hy": "paul",
    "hz": "george",
    "ex": "ringo",
    "ey": "the fifth beatle",
}


def get_channel_map(mt_system):
    if mt_system == "LEMI12":
        channel_map = LEMI_CHANNEL_MAP_12
    elif mt_system == "LEMI34":
        channel_map = LEMI_CHANNEL_MAP_34
    elif mt_system == "NIMS":
        channel_map = DEFAULT_CHANNEL_MAP
    elif mt_system == "beatles":
        channel_map = THE_BEATLES
    else:
        print(f"whoops mt_system {mt_system} unknown")
        channel_map = DEFAULT_CHANNEL_MAP
        # raise NotImplementedError
    return channel_map


def map_channels(mt_system):
    channel_map = get_channel_map(mt_system)
    ex = channel_map["ex"]
    ey = channel_map["ey"]
    hx = channel_map["hx"]
    hy = channel_map["hy"]
    hz = channel_map["hz"]
    return ex, ey, hx, hy, hz
