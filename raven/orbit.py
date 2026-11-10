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

    # calculate basis vectors
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

    # r = r_mag * rotate_axis_angle(e_unit, h_unit, params['true_anomaly'])
    # calculate v
    # v = cross(params['h'], params['eccentricity'] + r / r_mag) / params['p']

    return r, v

# FIXME: obtain the same elements used to calculate state
# def get_parameters(x:np.ndarray, v:np.ndarray, mu:float) -> dict:
#     """Given state (in some parent frame), returns orbital parameters.
    
#     The result is returned as: 
#     [
#     - specific angular momentum, h
#     - semi-major axis, a
#     - eccentricity, e
#     - inclination, i
#     - longitude of the ascending node, O
#     - argument of the periapsis, omega
#     ]
#     """

#     h = cross(x, v)
#     n = cross([0.0, 0.0, 1.0], h)

#     # handle non-inclined orbits
#     if norm(n) == 0:
#         n = np.array([1.0, 0.0, 0.0], float)

#     e = ((np.sum(v**2) - mu / norm(x))*x - np.dot(x, v)*v) / mu

#     params = {}
#     params.update({'h_vector':h})
#     params.update({'eccentricity_vector':e})
#     params.update({'h':norm(h)})
#     params.update({'eccentricity':norm(e)})
#     params.update({'p':np.sum(h**2) / mu})
#     params.update({'inclination':np.acos(h[2] / norm(h))})
#     params.update({'long_ascending':np.acos(n[0] / norm(n))})
#     params.update({'arg_periapsis':np.acos(np.dot(n, e) / (norm(n)*norm(e)))})
#     params.update({'true_anomaly':np.acos(np.dot(e, x) / (norm(e)*norm(x)))})

#     return params
