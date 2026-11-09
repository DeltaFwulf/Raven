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
    C = 10


# differentiate the state vector
def diff_state(h:float, x:np.ndarray, p:np.ndarray) -> np.ndarray:
    """Differentiates the state array."""

    state_len = 14
    pert_len = 11
    n = round(x.size / state_len)

    dx = np.zeros_like(x, float)
    for i in range(n):

        sa = i*state_len
        pa = i*pert_len
        ma = x[sa + State.M.value] + p[pa + Pert.MDOT.value]*h

        # velocity
        dx[sa + State.XDOT.value:sa + State.XDOT.value + 3] = x[sa + State.V.value:sa + State.V.value + 3]

        # rotation
        q = x[sa + State.Q.value:sa + State.Q.value + 4]
        omega = x[sa + State.W.value:sa + State.W.value + 3]
        dx[sa + State.Q.value:sa+ State.Q.value + 4] += 0.5*hamilton(q, np.r_[0, omega])

        I = np.diag(p[pa + Pert.I.value:pa + Pert.I.value + 3])
        torque = p[pa + Pert.T.value:pa + Pert.T.value + 3]
        dx[sa + State.W.value:sa + State.W.value + 3] += np.matmul(np.linalg.inv(I), torque - cross(omega, np.matmul(I, omega)))

        # uncoupled loads
        force = rotate_quaternion(p[pa + Pert.F.value:pa + Pert.F.value + 3], q*np.array([1, -1, -1, -1])) # get into universal frame
        dx[sa + State.A.value:sa + State.A.value + 3] = force / ma

        # change in mass
        dx[sa + State.M.value] = p[pa + Pert.MDOT.value]

        # coupled effects
        for j in range(n):
            pb = j*pert_len
            if i == j or p[j*pert_len + Pert.C.value] == 1:
                continue

            sb = j*state_len
            mb = x[sb + State.M.value] + p[pb + Pert.MDOT.value]*h

            # gravitational acceleration
            s = x[sa + State.X.value:sa + State.X.value + 3] - x[sb + State.X.value:sb + State.X.value + 3]
            dx[sa + State.A.value:sa + State.A.value + 3] -= 6.67408e-11*mb*s / norm(s)**3

            # aerodynamics
            # if ca == 1 and cb['atmosphere'] is not None:
            #     pass

    return dx


def rk4(x0:np.ndarray, perts:np.ndarray, h:float) -> np.ndarray:
    """Solves for new system state using RK4."""

    k1 = diff_state(0.0, x0, perts)
    k2 = diff_state(h / 2, x0 + k1*0.5*h, perts)
    k3 = diff_state(h / 2, x0 + k2*0.5*h, perts)
    k4 = diff_state(h, x0 + k3*h, perts)

    return x0 + h*(k1 + 2*(k2 + k3) + k4) / 6


def rk45(x0:np.ndarray, perts:np.ndarray, h_in:float, atol:float, rtol:float, hmax:float) -> tuple[np.ndarray, float]:
    """RK4 solver with adaptive timestep control."""

    delta = 2
    h_new = copy(h_in)

    converged = False
    while not converged:

        h = h_new
        # compute internal stages
        k1 = diff_state(0.0, x0, perts)
        k2 = diff_state(h / 4, x0 + 0.25*h*k1, perts)
        k3 = diff_state(3*h / 8, x0 + h*(3*k1 / 32 + 9*k2 / 32), perts)
        k4 = diff_state(h*12 / 13, x0 + (h / 2197)*(1932*k1 - 7200*k2 + 7296*k3), perts)
        k5 = diff_state(h, x0 + h*(439*k1 / 216 - 8*k2 + 3680*k3 / 513 - 845*k4 / 4104), perts)
        k6 = diff_state(h / 2, x0 + h*(-8*k1 / 27 + 2*k2 - 3544*k3 / 2565 + 1859*k4 / 4104 - 11*k5 / 40), perts)

        # compute 4th and 5th order estimates
        x4 = h*((25 / 216)*k1 + (1408 / 2565)*k3 + (2197/4104)*k4 - 0.2*k5)
        x5 = h*((16 / 135)*k1 + (6656 / 12825)*k3 + (28561 / 56430)*k4 - (9 / 50)*k5 + (2 / 55)*k6)

        # calculate tolerance and delta
        tol = atol + rtol*max(np.max(x4), np.max(x5))
        delta = np.linalg.norm((x5 - x4) / (tol*x0.size))
        r =min(5, delta**(-0.2))
        h_new = min(hmax, h*r)
        if delta <= 1:
            converged = True

    return x0 + x5, h, h_new


# def symplectic_4th_order(x:np.ndarray, v:np.ndarray, h:float, x_planet:np.ndarray, mu:float) -> None:
#     """Updates vehicle orbital position using a 4th-order symplectic integrator.
        
#     - Atmosphere is ignored here, use for long term orbital simulation."""

#     k = 2**(1/3)
#     c = 0.5 / (2 - k)*np.array([1, 1 - k, 1 - k, 1])
#     d = 1 / (2 - k)*np.array([1, -k, 1, 0])

#     for i in range(4):
#         a = -mu*(x - x_planet) / norm(x - x_planet)**3
#         v += h*c[i]*a
#         x += h*d[i]*v

#     return x, v
