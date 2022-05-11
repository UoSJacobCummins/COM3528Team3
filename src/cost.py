import math

def cost(rotations):
    return (30 * math.log(rotations + 2))

def cost_constant_stimulus(rotations, utility):
    if(rotations < 0):
        rotations = 0
    return utility / cost(rotations)
