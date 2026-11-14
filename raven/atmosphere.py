"""Contains atmospheric state functions."""

from math import exp, sqrt
import numpy as np


def coesa_low(z_km:float) -> np.ndarray:
    """Calculate atmospheric properties for Earth below 86 km."""

    # convert to geopotential alitude, h_km (clevor transformation to make press. linear)
    r0 = 6378.1
    h_km = r0*z_km / (r0 + z_km)
    temp, pres = 0.0, 0.0
    r_gas = 287.053

    # print(f"z_km: {z_km}, h_km: {h_km}")

    if h_km < 11: # Troposphere
        temp = 288.15 - 6.5*h_km
        pres = 101325.0*(288.15 / temp)**(-34.1632 / 6.5)
    elif h_km < 20:
        temp = 216.65
        pres = 22632.06*exp(-34.1632*(h_km - 11) / 216.65)
    elif h_km < 32:
        temp = 196.65 + h_km
        pres = 5474.889*(216.65 / temp)**34.1632
    elif h_km < 47:
        temp = 139.05 + 2.8*h_km
        pres = 868.0187*(228.65 / temp)**(34.1632 / 2.8)
    elif h_km < 51:
        temp = 270.65
        pres = 110.9063*exp(-34.1632*(h_km - 47) / 270.65)
    elif h_km < 71:
        temp = 413.45 - 2.8*h_km
        pres = 66.93887*(270 / temp)**(-34.1632 / 2.8)
    else:
        temp = 356.65 - 2.0*h_km
        pres = 3.95642*(214.65 / temp)**(-17.0816)

    dens = pres / (r_gas*temp)

    return np.array([temp, pres, dens], float)



def coesa_high(z_km:float) -> np.ndarray:
    """Calculate atmospheric properties for Earth between 86 and 1000 km."""

    state = np.zeros(3, float)

    if z_km < 91:
        state[0] = 186.8673
    elif z_km < 110:
        state[0] = 263.1905 - 76.3232*sqrt(1 - ((z_km - 91) / -19.9429)**2)
    elif z_km < 120:
        state[0] = 240 + 12*(z_km - 110)
    else:
        zeta = (z_km - 120)*(6356.766 + 120) / (6356.766 + z_km)
        state[0] = 1000 - 640*exp(-0.01875*zeta)

    if z_km < 91:
        kp = [0.0, 2.159582e-6, -4.836957e-4, -0.1425192, 13.4753]
        kd = [0.000000, -3.322622e-06, 9.111460e-04, -0.2609971, 5.944694]
    elif z_km < 100:
        kp = [0.0, 3.304895e-5, -0.009062730, 0.6516698, -11.03037]
        kd = [0.000000, 2.873405e-05, -0.008492037, 0.6541179, -23.62010]
    elif z_km < 110:
        kp = [0.0, 6.693926e-5, -0.01945388, 1.719080, -47.75030]
        kd = [-1.240774e-05, 0.005162063, -0.8048342, 55.55996, -1443.338]
    elif z_km < 120:
        kp = [0.0, -6.539316e-05, 0.02485568, -3.223620, 135.9355]
        kd = [0.00000, -8.854164e-05, 0.03373254, -4.390837, 176.5294]
    elif z_km < 150:
        kp = [2.283506e-07, -1.343221e-04, 0.02999016, -3.055446, 113.5764]
        kd = [3.661771e-07, -2.154344e-04, 0.04809214, -4.884744, 172.3597]
    elif z_km < 200:
        kp = [1.209434e-08, -9.692458e-06, 0.003002041, -0.4523015, 19.19151]
        kd = [1.906032e-08, -1.527799e-05, 0.004724294, -0.6992340, 20.50921]
    elif z_km < 300:
        kp = [8.113942e-10, -9.822568e-07, 4.687616e-04, -0.1231710, 3.067409]
        kd = [1.199282e-09, -1.451051e-06, 6.910474e-04, -0.1736220, -5.321644]
    elif z_km < 500:
        kp = [9.814674e-11, -1.654439e-07, 1.148115e-04, -0.05431334, -2.011365]
        kd = [1.140564e-10, -2.130756e-07, 1.570762e-04, -0.07029296, -12.89844]
    elif z_km < 750:
        kp = [-7.835161e-11, 1.964589e-07, -1.657213e-04, 0.04305869, -14.77132]
        kd = [8.105631e-12, -2.358417e-09, -2.635110e-06, -0.01562608, -20.02246]
    else:
        kp = [2.813255e-11, -1.120689e-07, 1.695568e-04, -0.1188941, 14.56718]
        kd = [-3.701195e-12, -8.608611e-09, 5.118829e-05, -0.06600998, -6.137674]

    state[1] = exp(sum(kp[4 - i]*z_km**i for i in range(0, 5)))
    state[2] = exp(sum(kd[4 - i]*z_km**i for i in range(0, 5)))

    return state



def coesa76(z:float) -> np.ndarray:
    """Returns atmospheric properties given altitude.
    
    Reference: https://www.braeunig.us/space/atmmodel.htm

    Output:
    [temperature (K), pressure (Pa), density (kg/m^3)]

    TODO: handle speed of sound when only can be obtained below 86km with this method.
    """

    z_km = z / 1000.0

    # bounds check
    if not 0 <= z_km <= 1000.0:
        return np.zeros(3, float)

    if z_km < 86:
        return coesa_low(z_km)

    return coesa_high(z_km)
