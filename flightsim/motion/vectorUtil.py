import numpy as np
from math import cos, sin, pi
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
    

    def transformLocal(self, translation:np.array=np.zeros((3),float), ang:float=0, axis:np.array=np.array([1,0,0],float)) -> None:
        """Transforms the current frame within its own local coordinate system"""

        # quaternion-based rotation matrix
        q = cos(ang / 2) * np.ones((4), float)
        q[1:] = sin(ang / 2) * axis

        # translation must first be rotated into the frame's reference frame (use the upper 3x3 matrix of current transformation matrix before applying)
        transGlobal = np.matmul(self.transform[:3,:3], translation)

        affineTransform = np.identity(4, float)
        affineTransform[:3,:3] = Transform.rotationMatrixFromQuaternion(q)
        affineTransform[:3, 3] = transGlobal

        self.transform = np.matmul(affineTransform, self.transform)


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


    def local2parent(self, vecIn:np.array) -> np.array:
        """Maps a vector defined relative to this reference frame into the parent frame's coordinate system"""
        return np.matmul(self.transform, np.append(vecIn, np.array([1])))[:-1]

       
    def parent2local(self, vecIn:np.array) -> np.array:
        """Maps a vector defined relative to the parent reference frame into this reference system"""
        # y = ax + b
        # a^-1(y-b) = x
        return np.matmul(self.transform[:3, :3].transpose(), vecIn - self.transform[:3, 3])
    

    def align(self, vecIn:np.array) -> np.array:
        """A pure rotation transformation that maps the input vector to a reference frame whose origin does not change but whose axes are now aligned with the local frame"""
        rotMat = self.transform[:3,:3]
        return np.matmul(rotMat, vecIn)
    

    def chain(self, prevTransform:'Transform'):
            """
            Chains transforms A (previous) and B (this one) together such that B is applied in A's local coordinate system.

            This is very similar to transform local (it does the same thing but takes in another transform for convenience)
            """

            self.transform = np.matmul(prevTransform.transform, self.transform)


    def getRotMatrix(self):
        return self.transform[:3, :3]
    

    def getTransVec(self):
        return self.transform[:3, -1]



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

        x = frame.local2parent(np.array([1,0,0]))
        y = frame.local2parent(np.array([0,1,0]))
        z = frame.local2parent(np.array([0,0,1]))
        o = frame.local2parent(np.array([0,0,0]))

        # update plot limits
        xMin = np.min(np.array([xMin, x[0], y[0], z[0]]))
        xMax = np.max(np.array([xMax, x[0], y[0], z[0]]))

        yMin = np.min(np.array([yMin, x[1], y[1], z[1]]))
        yMax = np.max(np.array([yMax, x[1], y[1], z[1]]))

        zMin = np.min(np.array([zMin, x[2], y[2], z[2]]))
        zMax = np.max(np.array([zMax, x[2], y[2], z[2]]))
        
        ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], '-r')
        ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], '-g')
        ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], '-b')

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