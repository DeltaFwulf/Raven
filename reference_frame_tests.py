import unittest
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from math import pi

from referenceFrame import ReferenceFrame



class testReferenceFrame(unittest.TestCase):


    def testFrame(self):
        
        # First, try to initialise a frame however is allowed

        # If this works, test movement of a trivial frame

        # If this works, chain two frames together

        # If this works, map vectors into and out of the frame with and excluding translation

        pass


    def test_init(self):
        pass


    def testMove(self):

        # create a reference frame aligned to the parent frame

        # translate the frame by (1, 1, 1) in the parent frame and local frame with origin (0,0,0) in local frame

        # now perform the same translation with origin (1, -1, 1) in local and parent frame

        # create a new frame with rotation of 90 degrees about the z axis

        # rotate the initial frame about the x axis by 45 degrees in the parent frame, oriin (0,0,0)

        # rotate the initial frame about the x axis by 45 degrees in the local frame, origin (0,0,0)

        # repeat but with origin (0, 0, 1) in both parent and local frames

        pass



def visualTest():

    frameA = ReferenceFrame()
    frameB = deepcopy(frameA)
    frameB.moveFrame(origin=np.array([1,0,0], float), axis=np.array([0,0,1], float), ang=30*pi / 180)
    frameC = deepcopy(frameB)
    frameC.moveFrame(origin=np.array([1,0,0], float), axis=np.array([1,0,0], float), ang=30*pi / 180, trans=np.array([-2, 0, 0], float), originFrame='parent', moveFrame='local')

    drawFrames([frameA, frameB, frameC])


def drawFrames(frames:list[ReferenceFrame]) -> None:
    """This function takes in a list of frames and for each one plots a set of orthogonal axes according to their respective transforms."""

    # TODO: add a text label next to each frame with sequence position
    ax = plt.figure().add_subplot(projection='3d')

    vmin = deepcopy(frames[0].origin)
    vmax = deepcopy(frames[-1].origin)

    for frame in frames:

        iUnit = frame.local2parent(np.array([1,0,0], float))
        jUnit = frame.local2parent(np.array([0,1,0], float))
        kUnit = frame.local2parent(np.array([0,0,1], float))
        o = frame.origin

        for i in range(3):
            vmin[i] = np.min(np.array([vmin[i], iUnit[i], jUnit[i], kUnit[i]]))
            vmax[i] = np.max(np.array([vmax[i], iUnit[i], jUnit[i], kUnit[i]]))
        
        ax.plot([o[0], iUnit[0]], [o[1], iUnit[1]], [o[2], iUnit[2]], '-r')
        ax.plot([o[0], jUnit[0]], [o[1], jUnit[1]], [o[2], jUnit[2]], '-g')
        ax.plot([o[0], kUnit[0]], [o[1], kUnit[1]], [o[2], kUnit[2]], '-b')

    ax.set_xlim([vmin[0], vmax[0]])
    ax.set_ylim([vmin[1], vmax[1]])
    ax.set_zlim([vmin[2], vmax[2]])
    ax.set_box_aspect([vmax[0] - vmin[0], vmax[1] - vmin[1], vmax[2] - vmin[2]])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(['x', 'y', 'z'])

    plt.show()

if __name__ == '__main__':
    visualTest()