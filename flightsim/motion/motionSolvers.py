import numpy as np



def linearRK4(x:np.array, v:np.array, h:float, accFn, params:dict) -> np.array:
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



def angularRK4(q0, omega0, h:float, rotFn, params:dict) -> np.array:

    def diff(dt:float, X:np.array, moments:function, mass) -> np.array:

        moment = moments(X, dt)
        
        Xdot = np.zeros(np.shape(X))
        Xdot[0,:] = X[1,:]
        Xdot[1,:] = 0

        return Xdot
    
    pass

    