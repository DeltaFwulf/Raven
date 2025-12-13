import numpy as np
from numpy.linalg import norm
from math import sin, cos, sqrt
from copy import deepcopy

from vectorUtil import getAngleUnsigned, rotateQuaternion, grassmann, unit, axisAngle2Quaternion

class ReferenceFrame():
    """
    This class represents the transformation between two reference frames. Use this class to map vectors or points between different axis systems.
    Also contains functions that can be used to map inertia tensors between axis systems according to the frame transform.

    Use this class to create reference frames, move reference frames, or map vectors into different frames of reference.
    """

    def __init__(self, axis:np.array=None, ang:float=None, translation:np.array=np.zeros(3,float), sphereAngs:np.array=None, roll:float=None, axisName:str=None):

        """
        Passing Spherical Coordinates with Roll:
            You can pass an array representing inclination and azimuth on the unit sphere (aligned to parent frame), with an explicit roll angle.
            In this case, the unit vector from the sphere origin and the location on the sphere is the new axis (named either 'x', 'y', or 'z').
            The transformation between the parent 'x', 'y', or 'z' is determined and sets the initial rotation matrix. The roll is then applied in the new frame's local system to get the final rotation matrix.
        """

        self.q = np.array([1,0,0,0], float)
        self.translation = translation

        if axis is not None:
            self.q = axisAngle2Quaternion(axis, ang)
        
        elif sphereAngs is not None:
            
            referenceAxis = np.zeros(3, float)

            match axisName:
                case 'x':
                    n = 0
                case 'y':
                    n = 1
                case 'z':
                    n = 2

            referenceAxis[n] = 1

            # what is the new axis?
            newAxis = np.zeros(3, float)
            newAxis[0] = sin(sphereAngs[0]) * cos(sphereAngs[1])
            newAxis[1] = sin(sphereAngs[0]) * sin(sphereAngs[1])
            newAxis[2] = cos(sphereAngs[0])

            # get the axis and angle of rotation
            if (newAxis == referenceAxis).all():
                rotAxis = np.array([1,0,0], float)
            else:
                rotAxis = np.cross(referenceAxis, newAxis)
                rotAxis /= norm(rotAxis) # normalise the vector
            
            angle = getAngleUnsigned(referenceAxis, newAxis)

            self.q = axisAngle2Quaternion(rotAxis, angle)
            self.move(referenceAxis, roll) # apply roll


    # # TODO: add the third mode, rotation within the parent frame (including rotation of translation vector)
    # def move(self, axis:np.array=np.array([1,0,0], float), ang:float=0, translation:np.array=np.array([0,0,0], float), reference:str='local') -> None:
    #     """Moves the reference frame according to a rotation and translation, in either local or parent frame's reference."""

    #     axis /= norm(axis) # normalise the axis (so that we can pass any axis of rotation into the function)

    #     if reference == 'parent':

    #         qConv = deepcopy(self.q)
    #         qConv[1:] *= -1.0
            
    #         axis = rotateQuaternion(axis, qConv)

    #         q = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
    #         q /= norm(q) # renormalise q here to prevent drift

    #         self.q = grassmann(self.q, q)
    #         self.translation += translation

    #     elif reference == 'local':

    #         q = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
    #         q /= norm(q) # prevent drift

    #         # chain the rotations, but apply the translation within the original frame
    #         translation = self.local2parent(translation, incTranslation=False)
    #         self.q = grassmann(self.q, q)
    #         self.translation += translation

    #     else:
    #         print("reference keyword invalid: please use either local or parent")


    def moveFrame(self, origin:np.array=np.zeros(3, float), axis:np.array=None, ang:float=None, trans:np.array=None, originFrame:str='local', moveFrame:str='parent') -> None:
        """This function moves the reference frame about a specified origin by a rotation and translation. The origin may be expressed in local or parent
           frames, as can the movement working frame."""
        
        if originFrame == 'local':
            origin = self.local2parent(origin, incTranslation=True)

        if moveFrame == 'local':
            axis = self.local2parent(axis, incTranslation=False)
            trans = self.local2parent(trans, incTranslation=True)

        qRot = np.array([1,0,0,0], float)

        if axis is not None:

            axis = unit(axis)
            if originFrame == 'local':
                axis = self.local2parent(axis, incTranslation=False)

            qRot = np.r_[cos(ang / 2), sin(ang / 2)*axis]
            self.q = unit(grassmann(qRot, self.q))

        if trans is not None:
            self.translation = origin + rotateQuaternion(self.translation - origin, qRot)

        # if frame == 'parent':
        #     if origin is not None:
        #         tRot = self.translation - origin
        #         qT = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
        #         self.translation = origin + rotateQuaternion(tRot, qT) + trans
        #     else:
        #         self.translation += trans

        #     qConv = deepcopy(self.q)
        #     qConv[1:] *= -1.0
        #     axis = rotateQuaternion(axis, qConv)

        #     qIn = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
        #     qIn /= norm(qIn) # renormalise q here to prevent drift

        #     self.q = grassmann(self.q, qIn)

        # elif frame == 'local':

        #     if origin is not None: 
        #         axisT = rotateQuaternion(axis, self.q)
        #         qT = np.array([cos(ang/2), axisT[0]*sin(ang/2), axisT[1]*sin(ang/2), axisT[2]*sin(ang/2)], float)

        #         op = self.local2parent(origin, incTranslation=True)
        #         tRot = self.translation - op
        #         self.translation = op + rotateQuaternion(tRot, qT)

        #     else:
        #         self.translation += self.local2parent(trans, incTranslation=False)

        #     qIn = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
        #     qIn /= norm(qIn) # prevent drift
        #     self.q = grassmann(self.q, qIn)

        # else:
        #     print(f"'{frame}' is not a valid keyword for moveAbout")
            

    def local2parent(self, vecIn:np.array, incTranslation:bool=True) -> np.array:
        """If the vector is described in the local coordinate system, this returns the same vector as expressed in the world coordinate system
        
        if incTranslation == False, this purely rotates the vector (assumes the vector passed in was from the parent frame origin)
        """
        
        vecOut = rotateQuaternion(vecIn, self.q)
        
        if incTranslation:
            vecOut += self.translation

        return vecOut

       
    def parent2local(self, vecIn:np.array, incTranslation:bool=True) -> np.array:
        """If the vector is specified in world coordinates, this is what the vector would be expressed as in local coordinates
        
        if incTranslation == False, this purely rotates the vector (assumes the localVector shared an origin with the local frame origin)
        """

        if incTranslation:
            vecIn -= self.translation

        qConv = deepcopy(self.q)
        qConv[1:] *= -1
        return rotateQuaternion(vecIn, qConv)
    

    def invert(self) -> None:
        """Inverts the transformation"""
        self.q[1:] *= -1.0
        self.translation = -self.translation

    
    def chain(self, nextFrame:'ReferenceFrame'):
            """This combines two subsequent transformations together, in the order of self.transform, newTransform"""
            self.q = grassmann(self.q, nextFrame.q)
            self.translation += nextFrame.translation


    def transformInertiaTensor(self, tensorIn, mass, com2ref:np.array=np.zeros(3)) -> np.array:
        """Changes the reference frame of the mass moment of inertia tensor to that about a reference frame with this transform from the body aligned, CoM centered reference frame.
        
        For generalised parallel axis theorem, you can also specify translation (in parent coordinates) of the reference location from the object's centre of mass. If the tensor input is about the object's centre of mass,
        do not put anything for initialTranslation. If the initial translation is non-zero, please put the translation in; the generalised parallel axis theorem method can take this into account.
        """
        
        rotating = self.q[0] != 1
        translating = (self.translation != np.zeros((3), float)).any()

        tensor = tensorIn

        if rotating:
            
            i = np.array([1, 0, 0])
            j = np.array([0, 1, 0])
            k = np.array([0, 0, 1])

            iNew = self.align(i)
            jNew = self.align(j)
            kNew = self.align(k)

            def cosAng(vec1, vec2):
                return np.dot(vec1, vec2) / sqrt(np.linalg.norm(vec1) * np.linalg.norm(vec2))
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

            def getSymmetricMatrix(a, b):

                c = np.empty((3,3), float)

                c[0,0] = a[1]*b[1] + a[2]*b[2]
                c[0,1] = -0.5 * (a[0]*b[1] + a[1]*b[0])
                c[0,2] = -0.5 * (a[0]*b[2] + a[2]*b[0])

                c[1,0] = c[0,1]
                c[1,1] = a[0]*b[0] + a[2]*b[2]
                c[1,2] = -0.5 * (a[1]*b[2] + a[2]*b[1])

                c[2,0] = c[0,2]
                c[2,1] = c[1,2]
                c[2,2] = a[0]*b[0] + a[1]*b[1]

                return c

            translation = self.translation

            tensor += (mass * getSymmetricMatrix(translation, translation)) - (2 * mass * getSymmetricMatrix(translation, com2ref))
                                                
            return tensor