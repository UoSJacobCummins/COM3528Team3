import math

def cost(rotations):
    return (30 * math.log(rotations + 2))

def cost_constant_stimulus(rotations, utility):
    return utility / cost(rotations)
