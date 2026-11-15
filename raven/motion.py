"""Contains motion solvers for Raven."""

from enum import Enum
from copy import copy
import numpy as np
from numpy.linalg import norm

from .vectors import hamilton, rotate_quaternion, cross


class State(Enum):
    """Indices within a body's state array."""
    X = 0
    XDOT = 0
    V = 3
    A = 3
    Q = 6
    W = 10
    M = 13


class Pert(Enum):
    """Indices within a body's property array."""
    F = 0
    T = 3
    I = 6
    MDOT = 9


# differentiate the state vector
def diff_state(h:float, x:np.ndarray, perts:np.ndarray, props:list[dict]) -> np.ndarray:
    """Differentiates the state array."""

    state_len = 14
    pert_len = 10
    n = round(x.size / state_len)

    dx = np.zeros_like(x, float)
    for i in range(n):

        sa = i*state_len
        pa = i*pert_len
        ma = x[sa + State.M.value] + perts[pa + Pert.MDOT.value]*h

        # velocity
        dx[sa + State.XDOT.value:sa + State.XDOT.value + 3] = x[sa + State.V.value:sa + State.V.value + 3]

        # rotation
        q = x[sa + State.Q.value:sa + State.Q.value + 4]
        omega = x[sa + State.W.value:sa + State.W.value + 3]
        dx[sa + State.Q.value:sa+ State.Q.value + 4] += 0.5*hamilton(q, np.r_[0, omega])

        I = np.diag(perts[pa + Pert.I.value:pa + Pert.I.value + 3])
        torque = perts[pa + Pert.T.value:pa + Pert.T.value + 3]
        dx[sa + State.W.value:sa + State.W.value + 3] += np.matmul(np.linalg.inv(I), torque - cross(omega, np.matmul(I, omega)))

        # uncoupled perturbations
        dx[sa + State.M.value] = perts[pa + Pert.MDOT.value]
        force = rotate_quaternion(perts[pa + Pert.F.value:pa + Pert.F.value + 3], q*np.array([1, -1, -1, -1])) # get into universal frame

        # coupled perturbations
        for j in range(n):
            pb = j*pert_len
            if i == j or props[j]['type'] == 1:
                continue

            sb = j*state_len
            mb = x[sb + State.M.value] + perts[pb + Pert.MDOT.value]*h

            # gravitational acceleration
            s = x[sa + State.X.value:sa + State.X.value + 3] - x[sb + State.X.value:sb + State.X.value + 3]
            s_mag = norm(s)
            dx[sa + State.A.value:sa + State.A.value + 3] -= 6.67408e-11*mb*s / s_mag**3

            # aerodynamics
            if props[i]['type'] == 1 and props[j]['type'] == 0 and props[j]['atmosphere'] is not None:                
                z = s_mag - props[j]['r']
                air = props[j]['atmosphere'](z)

                # freestream velocity in universal frame
                omega_parent = rotate_quaternion(x[sb + State.W.value:sb + State.W.value + 3],
                                                 x[sb + State.Q.value:sb + State.Q.value + 3])

                v_rel = x[sb + State.V.value:sb + State.V.value + 3] +\
                        cross(omega_parent, s) -\
                        x[sa + State.V.value:sa + State.V.value + 3]

                # TODO: better aero properties definition
                force += 0.5*air[2]*v_rel*norm(v_rel)*props[i]['cda']

        dx[sa + State.A.value:sa + State.A.value + 3] += force / ma

    return dx


# TODO: write unit tests
def rk45(x0:np.ndarray, perts:np.ndarray, props:list[dict], h_in:float, atol:float, rtol:float, hmax:float) -> tuple[np.ndarray, float, float]:
    """RK4 solver with adaptive timestep control."""

    delta = 2
    h_new = copy(h_in)

    while delta > 1:
        h = h_new
        # compute internal stages
        k1 = diff_state(0.0, x0, perts, props)
        k2 = diff_state(h / 4, x0 + 0.25*h*k1, perts, props)
        k3 = diff_state(3*h / 8, x0 + h*(3*k1 / 32 + 9*k2 / 32), perts, props)
        k4 = diff_state(h*12 / 13, x0 + (h / 2197)*(1932*k1 - 7200*k2 + 7296*k3), perts, props)
        k5 = diff_state(h, x0 + h*(439*k1 / 216 - 8*k2 + 3680*k3 / 513 - 845*k4 / 4104), perts, props)
        k6 = diff_state(h / 2, x0 + h*(-8*k1 / 27 + 2*k2 - 3544*k3 / 2565 + 1859*k4 / 4104 - 11*k5 / 40), perts, props)

        # compute 4th and 5th order estimates
        x4 = h*((25 / 216)*k1 + (1408 / 2565)*k3 + (2197/4104)*k4 - 0.2*k5)
        x5 = h*((16 / 135)*k1 + (6656 / 12825)*k3 + (28561 / 56430)*k4 - (9 / 50)*k5 + (2 / 55)*k6)

        # calculate tolerance and delta
        tol = atol + rtol*max(np.max(x4), np.max(x5))
        delta = np.linalg.norm((x5 - x4) / (tol*x0.size))
        r =min(5, delta**(-0.2))
        h_new = min(hmax, h*r)

    return x0 + x5, h, h_new
