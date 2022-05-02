import math


def battery_utility(charge):
    missing = 100-charge
    return math.exp(missing/20.629)


def cost(rotations, usage, time):
    return (usage * 30 * math.log(rotations + 2)) + time


def cost_constant_stimulus(rotations, utility, usage=1, time=0):
    return utility / cost(rotations, usage, time)


def cost_battery_stimulus(rotations, charge, usage=1, time=0):
    return battery_utility(charge) / cost(rotations, usage, time)
