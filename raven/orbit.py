"""Functions relating to orbital parameters, translating to state or solving the Kepler problem."""

import numpy as np

from .vectors import cross, rotate_axis_angle

def elements2state(elements:list, mu:float) -> tuple[np.ndarray, np.ndarray]:
    """Returns state vector (x,v) for a body, given orbital parameters."""

    rp = elements[0]
    e_mag = elements[1]
    inc = np.deg2rad(elements[2])
    long_asc = np.deg2rad(elements[3])
    arg_pe = np.deg2rad(elements[4])
    t_anom = np.deg2rad(elements[5])

    x = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    n = rotate_axis_angle(x, z, long_asc)
    h = rotate_axis_angle(z, n, inc)
    e_unit = rotate_axis_angle(n, h, arg_pe)
    q_unit = cross(h, e_unit)

    p = rp*(1 + e_mag)
    r_mag = p / (1 + e_mag*np.cos(t_anom))
    r = r_mag*(np.cos(t_anom)*e_unit + np.sin(t_anom)*q_unit)
    v = np.sqrt(mu / p)*((e_mag + np.cos(t_anom))*q_unit -np.sin(t_anom)*e_unit)

    return r, v
