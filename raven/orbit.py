"""Code relating to orbital calculations."""

import numpy as np
from numpy.linalg import norm

from .vectors import cross, rotate_axis_angle

def get_state(params:dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns state vector (x,v) for a body, given orbital parameters."""

    # calculate x
    r_mag = params['p'] / (1 + params['eccentricity']*np.cos(params['true_anomaly']))
    e_unit = params['eccentricity_vector'] / norm(params['eccentricity_vector'])
    h_unit = params['h_vector'] / norm(params['h_vector'])
    r = r_mag * rotate_axis_angle(e_unit, h_unit, params['true_anomaly'])
    
    # calculate v
    v = cross(params['h'], params['eccentricity'] + r / r_mag) / params['p']

    return r, v


def get_parameters(x:np.ndarray, v:np.ndarray, mu:float) -> dict:
    """Given state (in some parent frame), returns orbital parameters.
    
    The result is returned as: 
    [
    - specific angular momentum, h
    - semi-major axis, a
    - eccentricity, e
    - inclination, i
    - longitude of the ascending node, O
    - argument of the periapsis, omega
    ]
    """

    h = cross(x, v)
    n = cross([0.0, 0.0, 1.0], h)

    # handle non-inclined orbits
    if norm(n) == 0:
        n = np.array([1.0, 0.0, 0.0], float)

    e = ((np.sum(v**2) - mu / norm(x))*x - np.dot(x, v)*v) / mu

    params = {}
    params.update({'h_vector':h})
    params.update({'eccentricity_vector':e})
    params.update({'h':norm(h)})
    params.update({'eccentricity':norm(e)})
    params.update({'p':np.sum(h**2) / mu})
    params.update({'inclination':np.acos(h[2] / norm(h))})
    params.update({'long_ascending':np.acos(n[0] / norm(n))})
    params.update({'arg_periapsis':np.acos(np.dot(n, e) / (norm(n)*norm(e)))})
    params.update({'true_anomaly':np.acos(np.dot(e, x) / (norm(e)*norm(x)))})

    return params
