from math import exp




"""Contains atmospheric properties against different input conditions"""

def isa_atmo(z:float, dT:float=0):

    if z < 11000:

        T = (288.19 - 0.00649 * z) + dT
        P = 101290 * (T / 288)**5.526

    elif z >= 11000 and z < 25000:

        T = 216.69 + dT
        P = 22.56 * exp(1.73 - 0.000157 * z)

    else:

        T = (141.94 + 0.00299 * z) + dT
        P = 2.488 * (T / 216.6)**-11.388

    rho = P / (287 * T)

    return T, P, rho