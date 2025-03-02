import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt



class Transform():
    """
    This class represents the transformation between two reference frames. Use this class to map vectors or points between different axis systems.
    Also contains functions that can be used to map inertia tensors between axis systems according to the frame transform.

    Use this class to create reference frames, move reference frames, or map vectors into different frames of reference.


    Planned: 
    - accept euler angles for vehicle attitude
    """


    def __init__(self, transInit:np.array=np.zeros(3), angInit:float=0, axisInit:np.array=np.array([1,0,0], float), baseTransform:np.array=None):
        """The frame is initially defined by it rotation and translation relative to the world centre"""

        self.transform = np.identity(4, float)

        q = cos(angInit/2) * np.ones((4), float)
        q[1:] = sin(angInit / 2) * axisInit
        
        self.transform[:3,:3] = Transform.rotationMatrixFromQuaternion(q)
        self.transform[:3, 3] = transInit

        if baseTransform is not None:  # apply this transform to the base transform (chain):
            self.chain(baseTransform)


    def rotationMatrixFromQuaternion(q:np.array) -> np.array:
        """Given a unit quaternion, outputs a 3x3 rotation matrix"""

        rotMatrix = np.zeros((3,3), float)

        rotMatrix[0,0] = 2 * (q[0]**2 + q[1]**2) - 1
        rotMatrix[0,1] = 2 * (q[1]*q[2] - q[0]*q[3])
        rotMatrix[0,2] = 2 * (q[1]*q[3] + q[0]*q[2])
        
        rotMatrix[1,0] = 2 * (q[1]*q[2] + q[0]*q[3])
        rotMatrix[1,1] = 2 * (q[0]**2 + q[2]**2) - 1
        rotMatrix[1,2] = 2 * (q[2]*q[3] - q[0]*q[1])

        rotMatrix[2,0] = 2 * (q[1]*q[3] - q[0]*q[2])
        rotMatrix[2,1] = 2 * (q[2]*q[3] + q[0]*q[1])
        rotMatrix[2,2] = 2 * (q[0]**2 + q[3]**2) - 1

        return rotMatrix
    

    def move(self, axis:np.array=None, ang:float=None, translation:np.array=None, reference:str='local') -> None:
        """Moves the reference frame according to a rotation and translation, in either local or parent frame's reference."""

        transform = np.identity(4)

        if axis is not None:
            if reference == 'parent':
                axis = np.matmul(self.transform[:3,:3].transpose(), axis)
                
            q = np.zeros(4)
            q[0] = cos(ang / 2)
            q[1:] = sin(ang/2) * axis

            transform[:3, :3] = Transform.rotationMatrixFromQuaternion(q)

        if translation is not None:
            if reference == 'parent':
                translation = np.matmul(self.transform[:3,:3].transpose(), translation)

            transform[:3, 3] = translation

        self.transform = np.matmul(self.transform, transform)


    def invert(self) -> None:
        """Inverts the transformation"""
        self.transform[:3, :3] = self.transform[:3, :3].transpose()
        self.transform[:3, 3] = -self.transform[:3, 3]


    def local2parent(self, vecIn:np.array) -> np.array:
        """Maps a vector defined relative to this reference frame into the parent frame's coordinate system"""
        return np.matmul(self.transform, np.append(vecIn, np.array([1])))[:-1]

       
    def parent2local(self, vecIn:np.array) -> np.array:
        """Maps a vector defined relative to the parent reference frame into this reference system"""
        # y = ax + b,  a^-1(y-b) = x
        return np.matmul(self.transform[:3, :3].transpose(), vecIn - self.transform[:3, 3])
    

    def align(self, vecIn:np.array) -> np.array:
        """Maps a vector defined in this reference frame into the world frame BUT assumes that this reference frame shares an origin with the parent - pure rotation without translation"""
        return np.matmul(self.rotationMatrix(), vecIn)
    

    def chain(self, nextTransform:'Transform'):
            """This combines two subsequent transformations together, in the order of self.transform, newTransform"""

            self.transform = np.matmul(self.transform, nextTransform.transform)


    def rotationMatrix(self):
        return self.transform[:3, :3]
    

    def translation(self):
        return self.transform[:3, 3]
    

    def transformInertiaTensor(self, tensorIn, mass, com2ref:np.array=np.zeros(3)) -> np.array:
        """Changes the reference frame of the mass moment of inertia tensor to that about a reference frame with this transform from the body aligned, CoM centered reference frame.
        
        For generalised parallel axis theorem, you can also specify translation (in parent coordinates) of the reference location from the object's centre of mass. If the tensor input is about the object's centre of mass,
        do not put anything for initialTranslation. If the initial translation is non-zero, please put the translation in; the generalised parallel axis theorem method can take this into account.
        """
        
        rotating = (self.transform.rotationMatrix() != np.identity(3)).any()
        translating = (self.transform.translation() != np.zeros((3), float)).any()

        tensor = tensorIn

        if rotating:
            
            i = np.array([1, 0, 0])
            j = np.array([0, 1, 0])
            k = np.array([0, 0, 1])

            iNew = self.transform.align(i)
            jNew = self.transform.align(j)
            kNew = self.transform.align(k)

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

            translation = self.translation()

            tensor += (mass * getSymmetricMatrix(translation, translation)) - (2 * mass * getSymmetricMatrix(translation, com2ref))
                                             
        return tensor



def drawFrames(frames:list):

    """This function takes in a list of frames and for each one plots a set of orthogonal axes according to their respective transforms."""

    ax = plt.figure().add_subplot(projection='3d')

    xMin = 0
    xMax = 0
    yMin = 0
    yMax = 0
    zMin = 0
    zMax = 0

    for frame in frames:

        frameX = frame.local2parent(np.array([1,0,0]))
        frameY = frame.local2parent(np.array([0,1,0]))
        frameZ = frame.local2parent(np.array([0,0,1]))
        o = frame.local2parent(np.array([0,0,0]))

        # update plot limits
        xMin = np.min(np.array([xMin, frameX[0], frameY[0], frameZ[0]]))
        xMax = np.max(np.array([xMax, frameX[0], frameY[0], frameZ[0]]))

        yMin = np.min(np.array([yMin, frameX[1], frameY[1], frameZ[1]]))
        yMax = np.max(np.array([yMax, frameX[1], frameY[1], frameZ[1]]))

        zMin = np.min(np.array([zMin, frameX[2], frameY[2], frameZ[2]]))
        zMax = np.max(np.array([zMax, frameX[2], frameY[2], frameZ[2]]))
        
        ax.plot([o[0], frameX[0]], [o[1], frameX[1]], [o[2], frameX[2]], '-r')
        ax.plot([o[0], frameY[0]], [o[1], frameY[1]], [o[2], frameY[2]], '-g')
        ax.plot([o[0], frameZ[0]], [o[1], frameZ[1]], [o[2], frameZ[2]], '-b')

    # dynamically bound the plot based on the largest values of any terms in x, y, z
    ax.set_xlim([xMin, xMax])
    ax.set_ylim([yMin, yMax])
    ax.set_zlim([zMin, zMax])

    ax.set_box_aspect([xMax - xMin, yMax - yMin, zMax - zMin])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.legend(['x', 'y', 'z'])

    plt.show()