import numpy as np
from numpy.linalg import norm
from math import sqrt


class RigidBody():
        
    def __init__(self):
            
        self.mass = 0.0
        self.com = np.zeros(3, float)
        self.moi = np.zeros((3,3), float)


    def transformInertiaTensor(self, tensor:np.ndarray, mass:float, **kwargs) -> np.ndarray:
        """Changes the reference frame of the mass moment of inertia tensor to that about a reference frame with this transform from the body aligned, CoM centered reference frame.
        
        For generalised parallel axis theorem, you can also specify translation (in parent coordinates) of the reference location from the object's centre of mass. If the tensor input is about the object's centre of mass,
        do not put anything for initialTranslation. If the initial translation is non-zero, please put the translation in; the generalised parallel axis theorem method can take this into account.
        """

        rotating = self.q[0] != 1
        translating = (self.translation != np.zeros((3), float)).any()

        if rotating:
            
            i = np.array([1, 0, 0])
            j = np.array([0, 1, 0])
            k = np.array([0, 0, 1])

            iNew = self.align(i)
            jNew = self.align(j)
            kNew = self.align(k)

            def cosAng(vec1, vec2):
                return np.dot(vec1, vec2) / sqrt(norm(vec1) * norm(vec2))
            T = np.empty((3,3), float)
        
            T[0,0] = cosAng(iNew, i)
            T[0,1] = cosAng(iNew, j)
            T[0,2] = cosAng(iNew, k)

            T[1,0] = cosAng(jNew, i)
            T[1,1] = cosAng(jNew, j)
            T[1,2] = cosAng(jNew, k)

            T[2,0] = cosAng(kNew, i)
            T[2,1] = cosAng(kNew, j)
            T[2,2] = cosAng(kNew, k)

            tensor = np.matmul(T, np.matmul(tensor, np.transpose(T)))

        if translating:
            """
            This function uses a generalised form of the parallel axis theorem found here: https://doi.org/10.1119/1.4994835

            I' = Iref + M[(R2,R2)] - 2M[(R2,C)]
            """

            def getRelationalMatrix(veca, vecb):

                c = np.empty((3,3), float)

                c[0,0] = veca[1]*vecb[1] + veca[2]*vecb[2]
                c[0,1] = -0.5 * (veca[0]*vecb[1] + veca[1]*vecb[0])
                c[0,2] = -0.5 * (veca[0]*vecb[2] + veca[2]*vecb[0])

                c[1,0] = c[0,1]
                c[1,1] = veca[0]*vecb[0] + veca[2]*vecb[2]
                c[1,2] = -0.5 * (veca[1]*vecb[2] + veca[2]*vecb[1])

                c[2,0] = c[0,2]
                c[2,1] = c[1,2]
                c[2,2] = veca[0]*vecb[0] + veca[1]*vecb[1]

                return c

            translation = self.translation

            com2ref = np.zeros(3, float) if kwargs.get('com2ref') is None else kwargs.get('com2ref')

            tensor += (mass * getRelationalMatrix(translation, translation)) - (2 * mass * getRelationalMatrix(translation, com2ref))
                                                
            return tensor