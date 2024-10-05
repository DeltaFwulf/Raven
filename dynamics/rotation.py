import numpy as np
from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt



class Frame():
    # a frame stores its own translation between its parent frame and itself (rotation and translation)
    # this allows transformations to be calculated relative to this coordinate system and then mapped into the parent frame

    def __init__(self, transInit:np.array, angInit:float, axisInit:np.array):
        """The frame is initially defined by it rotation and translation relative to the world centre"""

        self.transform = np.identity(4, float)

        q = cos(angInit/2) * np.ones((4), float)
        q[1:] = sin(angInit / 2) * axisInit

        rotInit = np.array(([2*(q[0]**2 + q[1]**2) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
                                  [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]**2 + q[2]**2) - 1, 2*(q[2]*q[3] - q[0]*q[1])],
                                  [2*(q[1]*q[2] - q[0]*q[3]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]**2 + q[3]**2) - 1]), float)
        
        self.transform[:3,:3] = rotInit
        self.transform[:3, 3] = transInit



    def transform(self, transLocal:np.array, ang:float, axis:np.array) -> None:
        """
        this transformation is given in local coordinates. This must be performed before remapping into world coordinates.
        this should be possible by first applying the local transformation to a trivial vector, then applying the old transformation (intrinic operation)
        """

        # quaternion-based rotation matrix
        q = cos(ang / 2) * np.ones((4), float)
        q[1:] = sin(ang / 2) * axis

        rotMat = np.array(([2*(q[0]**2 + q[1]**2) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
                                  [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]**2 + q[2]**2) - 1, 2*(q[2]*q[3] - q[0]*q[1])],
                                  [2*(q[1]*q[2] - q[0]*q[3]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]**2 + q[3]**2) - 1]), float)

        affineLocal = np.identity(4, float)
        affineLocal[:3,:3] = rotMat
        affineLocal[:3, 3] = transLocal

        self.transform = np.matmul(affineLocal, self.transform)

        print(f"Local affine transformation matrix:\n{affineLocal}\n")
        print(f"world affine transformation matrix:\n{self.transform}\n")

        return
    

    def map(self, vecIn:np.array) -> np.array:
        """Applies the affine transformation matrix to an existing vector to map it into the local reference frame"""

        vecIn = np.vstack((vecIn, np.ones((1), float)))
        mapped = np.matmul(self.transform, vecIn)

        return mapped[:-1]
    


# There are two ways we can store the position of a frame:
# 1) store the initial transform and a second, updating transform that accounts for all subsequent transforms (full history is preserved)
# 2) store only the updating transform (less to store, but we don't know where we started or quickly see the effect of changing the initial position)
class Transform():
    """
    A set of ortholinear vectors used to describe 3d position.

    The frame is initialised relative to a parent set of axes, by rotating about the x,y, and finally z axes through angles
    alpha, beta, and gamma, respectively.

    The frame can be rotated using a unit quaternion, or by 3 rotations about its x, y, and z axes extrinsically.

    The current transformation matrix between parent axes [1;1;1] and Frame is stored in the class

    We update the vector transform relative to the frame, that is to say that we are in the frame's reference system when we apply transformations. 
    - The transform associated with the frame is to map back to a parent axis system XYZ. 

    Applying translations and rotations:
    - what order do we do this in
    - how to map translations back to origin


    Translations are applied at each timestep before the latest rotation; there are two methods that might work for this:
    - translating via a 3d vector in body frame transformed into world frame
    - making the spatial transformation matrix work to do this simultaneously
    """

    def __init__(self, alpha=0, beta=0, gamma=0, dx=0, dy=0, dz=0):

        self.transform = Transform.rotateExtrinsic("xyz", alpha, beta, gamma)

    def rotateExtrinsic(order, alpha, beta, gamma):
        """Returns the rotation matrix obtained when 3 successive extrinsic rotations are 
        performed about the frame's axes"""

        def rotateX(alpha):
            return np.array(([1, 0, 0], [0, cos(alpha), -sin(alpha)],[0, sin(alpha), cos(alpha)]))
        
        def rotateY(beta):
            return np.array(([cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]))
        
        def rotateZ(gamma):
            return np.array(([cos(gamma), -sin(gamma), 0],[sin(gamma), cos(gamma), 0], [0, 0, 1]))
        

        funcs = {'x':rotateX, 'y':rotateY, 'z':rotateZ}
        angs = {'x':alpha, 'y':beta, 'z':gamma}

        transform = np.identity(3)

        for axis in order:
            transform = np.matmul(funcs[axis](angs[axis]), transform)

        return transform
    

    def rotateQuaternion(self, q) -> None:

        rotationMatrix = np.array(([2*(q[0]**2 + q[1]**2) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
                                  [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]**2 + q[2]**2) - 1, 2*(q[2]*q[3] - q[0]*q[1])],
                                  [2*(q[1]*q[2] - q[0]*q[3]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]**2 + q[3]**2) - 1]), float)
        
        self.transform = np.matmul(self.transform, rotationMatrix) # intrinsic rotation (we are rotating in the frame of reference of frame)


    def putInWorldFrame(self, vector):
        return np.matmul(self.transform, vector)




def drawFrames(frames:tuple):

    """Draws the tuple of frames in a 3d plot"""
    ax = plt.figure().add_subplot(projection='3d')

    for frame in frames:

        x = frame.putInWorldFrame(np.array([1,0,0]))
        y = frame.putInWorldFrame(np.array([0,1,0]))
        z = frame.putInWorldFrame(np.array([0,0,1]))
        o = frame.putInWorldFrame(np.array([0,0,0]))
        
        ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], '-r')
        ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], '-g')
        ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], '-b')

        ax.set_box_aspect([1,1,1])

    plt.show()
        

# gather frame behaviour in this testbed, use findings to define a useful transformer class
def frameTest():

    # set up a world frame and local frame

    # transform the local frame relative to the world frame

    # attempt to map a set of axes into the local frame

    # plot the two frames relative to the world frame

    pass
        
frameTest()


