import numpy as np
from copy import deepcopy

from .referenceFrame import ReferenceFrame



class RigidBody():
        
    def __init__(self):
            
        self.mass = 0.0
        self.com = np.zeros(3, float)
        self.moi = np.zeros((3,3), float)


    def transformInertiaTensor(self, frame:ReferenceFrame, **kwargs) -> np.ndarray:
        """This function allows the moment of inertia of a rigid body to be expressed in a new reference frame, given a frame of reference relative to the rigid body's root frame.
           
           Unless specified, the function assumes that I is about the body's centre of mass. To specify a different initial I reference point, a keyword argument, 'ref' may be input with the vector
           from the object's original root frame to the reference point within the body local frame."""
        
        ref = self.com if kwargs.get('ref') is None else kwargs.get('ref') # the point about which moi was calculated, in root frame
        r = frame.parent2local(-ref, incTranslation=True) # from initial reference to frame origin
        c = frame.parent2local(self.com - ref, incTranslation=False) # from initial reference to centre of mass
        I = deepcopy(self.moi)

        if np.any(frame.q != np.array([1, 0, 0, 0], float)):
            
            i = np.array([1, 0, 0], float)
            j = np.array([0, 1, 0], float)
            k = np.array([0, 0, 1], float)

            u = frame.local2parent(np.array([1, 0, 0], float), incTranslation=False)
            v = frame.local2parent(np.array([0, 1, 0], float), incTranslation=False)
            w = frame.local2parent(np.array([0, 0, 1], float), incTranslation=False)

            T = np.zeros((3,3), float)

            T[0,0] = np.dot(i, u)
            T[0,1] = np.dot(j, u)
            T[0,2] = np.dot(k, u)

            T[1,0] = np.dot(i, v)
            T[1,1] = np.dot(j, v)
            T[1,2] = np.dot(k, v)

            T[2,0] = np.dot(i, w)
            T[2,1] = np.dot(j, w)
            T[2,2] = np.dot(k, w)

            I = np.matmul(T, np.matmul(I, np.transpose(T)))

        if np.any(r != np.zeros(3, float)):
            """This function uses a generalised form of the parallel axis theorem found here: https://doi.org/10.1119/1.4994835"""


            def getRelationalMatrix(a:np.ndarray, b:np.ndarray) -> np.ndarray:

                M = np.zeros((3,3), float)

                M[0,0] = a[1]*b[1] + a[2]*b[2]
                M[0,1] = -0.5*(a[0]*b[1] + a[1]*b[0])
                M[0,2] = -0.5*(a[0]*b[2] + a[2]*b[0])

                M[1,0] = M[0,1]
                M[1,1] = a[0]*b[0] + a[2]*b[2]
                M[1,2] = -0.5*(a[1]*b[2] + a[2]*b[1])

                M[2,0] = M[0,2]
                M[2,1] = M[1,2]
                M[2,2] = a[0]*b[0] + a[1]*b[1]

                return M
            

            I += self.mass*(getRelationalMatrix(r, r) - 2*getRelationalMatrix(r, c))
                                                
        return I