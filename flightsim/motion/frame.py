import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt



class Transform():
    """
    This class represents the transformation between two reference frames. Use this class to map vectors or points between different axis systems.
    
    Also contains functions that can be used to map inertia tensors between axis systems according to the frame transform.
    """

    # TODO: move inertia tensor reference move functions into this class, remove from Primitive class

    def __init__(self, transInit:np.array, angInit:float, axisInit:np.array):
        """The frame is initially defined by it rotation and translation relative to the world centre"""

        self.transform = np.identity(4, float)

        q = cos(angInit/2) * np.ones((4), float)
        q[1:] = sin(angInit / 2) * axisInit
        
        self.transform[:3,:3] = Transform.rotationMatrixFromQuaternion(q)
        self.transform[:3, 3] = transInit

        self.rotMatrix = self.transform[:3,:3]
        self.transVector = self.transform[:3, 3]


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
    

    def transformLocal(self, transLocal:np.array, ang:float, axis:np.array) -> None:
        """Transforms the current frame within its own local coordinate system"""

        # quaternion-based rotation matrix
        q = cos(ang / 2) * np.ones((4), float)
        q[1:] = sin(ang / 2) * axis

        # translation must first be rotated into the frame's reference frame (use the upper 3x3 matrix of current transformation matrix before applying)
        transGlobal = np.matmul(self.transform[:3,:3], transLocal)

        affineTransform = np.identity(4, float)
        affineTransform[:3,:3] = Transform.rotationMatrixFromQuaternion(q)
        affineTransform[:3, 3] = transGlobal

        self.transform = np.matmul(affineTransform, self.transform)

        self.rotMatrix = self.transform[:3, :3]
        self.transVector = self.transform[:3, 3]

        return
    

    def map(self, vecIn:np.array) -> np.array:
        """Applies the affine transformation matrix to an existing vector to map it into the local reference frame"""

        vecIn = np.append(vecIn, np.array([1]))
        mapped = np.matmul(self.transform, vecIn)

        return mapped[:-1] # reshaping in this function is unnecessary
    

    def align(self, vecIn:np.array) -> np.array:
        """Pure rotation transformation part of the affine transformation. This is equivalent to mapping the vector to a reference system whose origin still lies at the world origin, but whose axes are parallel to the local frame's"""
        rotMat = self.transform[:3,:3]
        return np.matmul(rotMat, vecIn)



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

        x = frame.map(np.array([1,0,0]))
        y = frame.map(np.array([0,1,0]))
        z = frame.map(np.array([0,0,1]))
        o = frame.map(np.array([0,0,0]))

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

    plt.show()
        

# gather frame behaviour in this testbed, use findings to define a useful transformer class
def frameTest():

    worldFrame = Transform(np.array([0,0,0]), 0, np.array([1,0,0]))
    
    # The launch rail is tilted at 5 degrees from the vertical, facing in the east direction (+x bias of z vector). Therefore, the base rotation is -5 degrees about the axis [0,1,0]
    railTilt = 30 * pi / 180
    railNorm = np.array([0,1,0], float)

    transInit = np.array([1,0,0])

    vehFrame = Transform(transInit, railTilt, railNorm)
    vehFrame.transformLocal(np.array([5,5,0]), -30*pi/180, np.array([0,0.5,0.5]))

    print(f"vehicle frame transformation matrix:\n{vehFrame.transform}\n")

    # plot both frames:
    frames = [worldFrame, vehFrame]
    drawFrames(frames)