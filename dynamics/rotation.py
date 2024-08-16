import numpy as np
from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt


# There are two ways we can store the position of a frame:
# 1) store the initial transform and a second, updating transform that accounts for all subsequent transforms (full history is preserved)
# 2) store only the updating transform (less to store, but we don't know where we started or quickly see the effect of changing the initial position)



# FIXME: make all functions consistent, the two rotation methods have different ideas of what they return
class Frame():
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
    """

    def __init__(self, alpha=0, beta=0, gamma=0, dx=0, dy=0, dz=0):

        self.transform = Frame.rotateExtrinsic("xyz", alpha, beta, gamma)

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


    def transformVector(self, vector):
        """maps a vector that is known relative to bodyFrame into worldFrame coordinates"""
        return np.matmul(self.transform, vector)




def drawFrames(frames:tuple):

    """Draws the tuple of frames in a 3d plot"""
    ax = plt.figure().add_subplot(projection='3d')

    for frame in frames:

        x = frame.transformVector(np.array([1,0,0]))
        y = frame.transformVector(np.array([0,1,0]))
        z = frame.transformVector(np.array([0,0,1]))
        o = frame.transformVector(np.array([0,0,0]))
        
        ax.plot([o[0], x[0]], [o[1], x[1]], [0[2], x[2]], '-r')
        ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], '-g')
        ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], '-b')

    
    plt.show()
        



def frameTest():

    # create a starting frame
    worldFrame = Frame()
    bodyFrame = Frame(alpha=0, beta=0, gamma=0)

    # let's rotate bodyFrame in its own reference system, getting the new transform to world frame after the rotation:
    # we'll assume we've computed the angular velocity vector and so can apply a quaternion transformation to the reference system:
    omega = 2 * np.array([1, 0, 0])
    timestep = 1 # second

    # express the angular velocity as a quaternion:
    dTheta = sqrt(np.sum(omega**2)) * timestep
    wOrientation = omega / sqrt(np.sum(omega**2))
    
    wQuaternion = cos(dTheta / 2) * np.ones((4), float)
    wQuaternion[1:] = sin(dTheta / 2) * wOrientation

    bodyFrame.rotateQuaternion(wQuaternion)

    print(f"transform after quaternion rotation:\n{bodyFrame.transform}")
    
    #drawFrames((worldFrame, bodyFrame))

frameTest()


