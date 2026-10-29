
import numpy as np
from numpy.linalg import norm

from .vectors import v_rotation

# TODO: Calculate relative velocity within vectors, not planet.
# TODO: incorporate rotation to rk4

def rk4(x_rocket:np.ndarray, x_planet:np.ndarray, h:float, m:float, mu:float, r_planet:float, cda:float=0.0, dm:float=0.0, atmo_model:str='none') -> np.ndarray:
    """Updates body state using RK4.
    
    State vectors (x) are given in the form: (x_x, x_y, x_z, v_x, v_y, v_z, ω_x, ω_y, ω_z)

    Assumptions:
    - gravity from one planet only
    - state vectors expressed in a common frame
    """

    def diff(x:np.ndarray, m:float) -> np.ndarray:
        """Differentiates the state vector.

        returns dx, of form: (v_x, v_y, v_z, a_x, a_y, a_z)
        for now, rotation is neglected.
        """

        dx = np.zeros(6, float)
        dx[:3] = x[3:6]

        # spherical gravity acceleration
        r = x_rocket[:3] - x_planet[:3]
        dx[3:6] = -mu*r / norm(r)**3

        if atmo_model != 'none':
            v_rel = x[3:6] - x_planet[3:6] - v_rotation(r, x_planet[6:9])
            rho = 0.0
            dx[3:6] -= 0.5*rho*norm(v_rel)*v_rel*cda / m

        # TODO: include rotation

        return dx

    # XXX: verify dm implementation (not sure)
    k1 = diff(x_rocket, m)
    k2 = diff(x_rocket + 0.5*k1*h, m + 0.5*h*dm)
    k3 = diff(x_rocket + 0.5*k2*h, m + 0.5*h*dm)
    k4 = diff(x_rocket + k3*h, m + h*dm)

    return x_rocket + (h / 6)*(k1 + 2*(k2 + k3) + k4)


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
