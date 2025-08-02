import numpy as np
from utility.vectorUtil import grassmann
from copy import deepcopy



def linearRK4(x:np.array, v:np.array, h:float, accFn, params:dict):
    """
    Explicit Runge-Kutta-Nyström [4] method to solve vehicle linear position and velocity.
    """


    def diff(dt:float, X:np.array, accFunction, params:dict) -> np.array:
        """Differentiates the current state equation"""

        Xdot = np.zeros(np.shape(X))
        Xdot[0,:] = X[1,:]
        Xdot[1,:] = accFunction(X, dt, params)

        return Xdot
    

    X0 = np.vstack((x,v))
    
    k1 = diff(0, X0, accFn, params)
    k2 = diff(h/2, X0 + (k1 * h / 2), accFn, params)
    k3 = diff(h/2, X0 + (k2 * h / 2), accFn, params)
    k4 = diff(h, X0 + k3 * h, accFn, params)

    X1 = X0 + (h/6) * (k1 + 2*(k2 + k3) + k4)

    return X1[0,:], X1[1,:]



def angularRK4(q0:np.array, omega0:np.array, I:np.array, t0:float, h:float, torqueFn):

    """Given a shape with known inertia tensor, centre of mass, moments, and initial state, calculates the angular motion over a given timestep using Runge-Kutta Nystrom 4 Algorithm.
    
    A function must be supplied that calculates the torque on a body given its angle, angular velocity, position, velocity, etc (torque is calculated in the body frame). For testing, the torque will be set to 0.
    
    """

    # TODO: allow I to change throughout a timestep (fuel burn, etc) using linear interpolation

    def qAcc(q, qDot, t, I, torqueFn):

        qConj = deepcopy(q)
        qConj[1:] *= -1 

        omega = 2 * grassmann(qConj, qDot)[1:] # vector
        torque = torqueFn(t, q, omega) # pseudovector

        A = grassmann(grassmann(qDot, qConj), qDot) # quaternion
        B = np.matmul(np.linalg.inv(I), torque - 4*np.cross(0.5*omega, np.matmul(I, 0.5*omega))) # vector
        C = 0.5 * grassmann(q, np.hstack((0, B))) # quaternion

        return A + C # quaternion
        

    qDot0 = 0.5 * grassmann(q0, np.hstack((0, omega0)))
    
    k1 = qAcc(q0, qDot0, t0, I, torqueFn)
    k2 = qAcc(q0 + (0.5*h*qDot0) + (h**2 / 8 * k1), qDot0 + 0.5*h*k1, t0 + h/2, I, torqueFn)
    k3 = qAcc(q0 + (0.5*h*qDot0) + (h**2 / 8 * k2), qDot0 + 0.5*h*k2, t0 + h/2, I, torqueFn)
    k4 = qAcc(q0 + h*qDot0 + (h**2 / 2 * k3), qDot0 + h*k3, t0 + h, I, torqueFn)

    q1 = q0 + h*qDot0 + (h**2 / 6)*(k1 + k2 + k3)
    qDot1 = qDot0 + (h/6)*(k1 + 2*(k2 + k3) + k4)

    qConj1 = deepcopy(q1)
    qConj1[1:] *= -1

    omega1 = 2 * grassmann(qConj1, qDot1)[1:]

    return q1, omega1