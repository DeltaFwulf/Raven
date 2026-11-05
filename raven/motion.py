"""Contains motion solvers for Raven."""

from enum import Enum
import numpy as np
from numpy.linalg import norm

from .vectors import hamilton, rotate_quaternion

class State(Enum):
    """Indices within a body's state array."""
    X = 0
    XDOT = 0
    V = 3
    A = 3
    Q = 6
    W = 10
    M = 13

class Prop(Enum):
    """Indices within a body's property array."""
    F = 0
    T = 3
    I = 6
    MDOT = 9
    C = 10

class BodyTypes(Enum):
    """Body class codes."""
    ROCKET = 1


def rk4(state_in:np.ndarray, props:np.ndarray, h:float) -> np.ndarray:
    """Solves for new system state using RK4."""

    state_len = 14
    pert_len = 11
    n = round(state_in.size / state_len)

    # differentiate the state vector
    def diff_state(h:float, x:np.ndarray, p:np.ndarray) -> np.ndarray:
        """Differentiates the state array."""

        dx = np.zeros_like(x, float)

        # FIXME: sum cumulative effects
        # FIXME: calculate for body A, only calculates once per object in this case

        for i in range(n):

            sa = i*state_len
            pa = i*pert_len
            ca = p[pa + Prop.C.value]
            ma = x[sa + State.M.value] + p[pa + Prop.MDOT.value]*h

            # velocity
            dx[sa + State.XDOT.value:sa + State.XDOT.value + 3] = x[sa + State.V.value:sa + State.V.value + 3]

            # rotation
            q = x[sa + State.Q.value:sa + State.Q.value + 4]
            omega = x[sa + State.W.value:sa + State.W.value + 3]
            dx[sa + State.Q.value:sa+ State.Q.value + 4] += 0.5*hamilton(q, np.r_[0, omega])

            I = np.diag(p[pa + Prop.I.value:pa + Prop.I.value + 3])
            torque = p[pa + Prop.T.value:pa + Prop.T.value + 3]
            dx[sa + State.W.value:sa + State.W.value + 3] += np.matmul(np.linalg.inv(I), torque - np.cross(omega, np.matmul(I, omega)))

            # uncoupled loads
            force = rotate_quaternion(p[pa + Prop.F.value:pa + Prop.F.value + 3], q*np.array([1, -1, -1, -1])) # get into universal frame
            dx[sa + State.A.value:sa + State.A.value + 3] = force / ma

            # change in mass
            dx[sa + State.M.value] = p[pa + Prop.MDOT.value]

            # coupled effects
            for j in range(n):

                if i == j:
                    continue

                # We calculate body b's diff vector
                sb = j*state_len
                pb = j*pert_len
                cb = p[pb + Prop.C.value]
                mb = x[sb + State.M.value] + p[pb + Prop.MDOT.value]*h

                # gravitational acceleration
                s = x[sa + State.X.value:sa + State.X.value + 3] - x[sb + State.X.value:sb + State.X.value + 3]
                dx[sa + State.A.value:sa + State.A.value + 3] -= 6.67408e-11*mb*s / norm(s)**3

                # aerodynamics
                if ca == 0 and cb == BodyTypes.ROCKET.value:
                    pass

        return dx

    k1 = diff_state(0.0, state_in, props)
    k2 = diff_state(h / 2, state_in + k1*0.5*h, props)
    k3 = diff_state(h / 2, state_in + k2*0.5*h, props)
    k4 = diff_state(h, state_in + k3*h, props)

    return state_in + h*(k1 + 2*(k2 + k3) + k4) / 6


def symplectic_4th_order(x:np.ndarray, v:np.ndarray, h:float, x_planet:np.ndarray, mu:float) -> None:
    """Updates vehicle orbital position using a 4th-order symplectic integrator.
    
    NOTE: for now, I do not know how to incorporate rotations.
    
    - Atmosphere is ignored here, use for long term orbital simulation."""

    k = 2**(1/3)
    c = 0.5 / (2 - k)*np.array([1, 1 - k, 1 - k, 1])
    d = 1 / (2 - k)*np.array([1, -k, 1, 0])

    for i in range(4):
        a = -mu*(x - x_planet) / norm(x - x_planet)**3
        v += h*c[i]*a
        x += h*d[i]*v

    return x, v


# def angularRK4(q0:np.array, omega0:np.array, I:np.array, t0:float, h:float, torqueFn):

#     """Given a shape with known inertia tensor, centre of mass, moments, and initial state, calculates the angular motion over a given timestep using Runge-Kutta Nystrom 4 Algorithm.

#     A function must be supplied that calculates the torque on a body given its angle, angular velocity, position, velocity, etc (torque is calculated in the body frame). For testing, the torque will be set to 0.
#     """

#     def qAcc(q, qDot, t, I, torqueFn):

#         qConj = deepcopy(q)
#         qConj[1:] *= -1

#         omega = 2 * grassmann(qConj, qDot)[1:] # vector
#         torque = torqueFn(t, q, omega) # pseudovector

#         A = grassmann(grassmann(qDot, qConj), qDot) # quaternion
#         B = np.matmul(np.linalg.inv(I), torque - 4*np.cross(0.5*omega, np.matmul(I, 0.5*omega))) # vector
#         C = 0.5 * grassmann(q, np.hstack((0, B))) # quaternion

#         return A + C # quaternion


#     qDot0 = 0.5 * grassmann(q0, np.hstack((0, omega0)))

#     k1 = qAcc(q0, qDot0, t0, I, torqueFn)
#     k2 = qAcc(q0 + (0.5*h*qDot0) + (h**2 / 8 * k1), qDot0 + 0.5*h*k1, t0 + h/2, I, torqueFn)
#     k3 = qAcc(q0 + (0.5*h*qDot0) + (h**2 / 8 * k2), qDot0 + 0.5*h*k2, t0 + h/2, I, torqueFn)
#     k4 = qAcc(q0 + h*qDot0 + (h**2 / 2 * k3), qDot0 + h*k3, t0 + h, I, torqueFn)

#     q1 = q0 + h*qDot0 + (h**2 / 6)*(k1 + k2 + k3)
#     qDot1 = qDot0 + (h/6)*(k1 + 2*(k2 + k3) + k4)

#     qConj1 = deepcopy(q1)
#     qConj1[1:] *= -1

#     omega1 = 2 * grassmann(qConj1, qDot1)[1:]

#     return q1, omega1
