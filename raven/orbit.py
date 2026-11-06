"""Code relating to orbital calculations."""

import numpy as np


def get_state(parameters) -> np.ndarray:
    """Returns state vector (x,v) for a body, given orbital parameters."""

    state = np.zeros(6, float)

    # calculate x

    # calculate v


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
    params = {}
    params.update({'h':np.cross(x, v)})

    return params
