import unittest
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from math import pi, sin, cos
from numpy.testing import assert_allclose, assert_array_equal
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raven.referenceFrame import ReferenceFrame
from raven.vectorUtil import grassmann, qRotate


# NOTE: to trust these test results, vector utils test cases must all have passed first since last change
#       it is recommended to run these tests within a suite, after the vector util tests

class FrameTests(unittest.TestCase):


    def testInit(self): # TODO: check for q and origin creation
        frame = ReferenceFrame()
        assert_array_equal(frame.q, np.array([1,0,0,0], float))
        assert_array_equal(frame.origin, np.zeros(3, float))


    def testAxisAngle(self):

        axis = np.array([1, 0, 0], float)
        ang = pi / 4
        origin = np.array([4, 0, 0], float)

        expectedQ = np.r_[cos(ang / 2), sin(ang / 2)*axis / np.linalg.norm(axis)]

        frame = ReferenceFrame()
        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        assert_allclose(frame.q, expectedQ, atol=1e-9)
        assert_allclose(frame.origin, origin, atol=1e-9)


    def testAxisAngle_bigAngle(self):

        axis = np.array([1, 0, 0], float)
        ang = 9*pi / 4
        origin = np.array([4, 0, 0], float)

        expectedQ = np.r_[cos(pi / 8), sin(pi / 8)*axis / np.linalg.norm(axis)]

        frame = ReferenceFrame()
        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        assert_allclose(frame.q, expectedQ, atol=1e-9)
        assert_allclose(frame.origin, origin, atol=1e-9)


    def testAxisAngle_negAngle(self):

        axis = np.array([1, 0, 0], float)
        ang = -7*pi / 4
        origin = np.array([4, 0, 0], float)

        expectedQ = np.r_[cos(pi / 8), sin(pi / 8)*axis / np.linalg.norm(axis)]

        frame = ReferenceFrame()
        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        assert_allclose(frame.q, expectedQ, atol=1e-9)
        assert_allclose(frame.origin, origin, atol=1e-9)


    def testAxisAngle_bigAxis(self):

        axis = np.array([20, -1, 0], float)
        ang = pi / 4
        origin = np.array([4, 0, 0], float)

        expectedQ = np.r_[cos(pi / 8), sin(pi / 8)*axis / np.linalg.norm(axis)]

        frame = ReferenceFrame()
        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        assert_allclose(frame.q, expectedQ, atol=1e-9)
        assert_allclose(frame.origin, origin, atol=1e-9)

    
    def testAxisAngle_nullAxis(self):
        
        axis = np.zeros(3, float)
        ang = pi / 4
        origin = np.array([4, 0 ,0], float)

        frame = ReferenceFrame()

        with self.assertRaises(ValueError):
            frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)


    def test_placeBaseVectors_oneVector(self):

        frame = ReferenceFrame()
        frame.placeBaseVectors(seq = ['x'], vectors=[np.array([cos(pi / 4), 0, sin(pi / 4)], float)])

        q_exp = np.r_[cos(pi / 8), 0, -sin(pi / 8), 0]
        o_exp = np.zeros(3, float)

        assert_allclose(frame.q, q_exp, atol=1e-9)
        assert_allclose(frame.origin, o_exp, atol=1e-9)

    
    def test_placeBaseVectors_twoVectors(self):

        frame = ReferenceFrame()

        vecA = np.array([0, 1, 0], float)
        vecB = np.array([1, 0, 0], float)

        frame.placeBaseVectors(seq = ['x', 'y'], vectors=[vecA, vecB])
   
        assert_allclose(frame.q, np.r_[0, np.array([cos(pi/4), sin(pi/4), 0], float)], atol=1e-9)
        assert_allclose(frame.origin, np.zeros(3, float), atol=1e-9)


    def test_placeBaseVectors_parallelToTarget(self):
        
        frame = ReferenceFrame()

        vecA = np.array([1, 0, 0], float)
        vecB = np.array([0, 1, 0], float)

        frame.placeBaseVectors(seq=['x', 'y'], vectors=[vecA, vecB])

        assert_allclose(frame.q, np.array([1, 0, 0, 0], float), atol=1e-9)
        assert_allclose(frame.origin, np.zeros(3, float), atol=1e-9)


    def test_placeBaseVectors_acuteAngle(self):

        frame = ReferenceFrame()

        vecA = np.array([0, 1, 0], float)
        vecB = np.array([100, 1, 0], float)

        frame.placeBaseVectors(seq=['x','y'], vectors=[vecA, vecB])
        
        expectedQ = np.r_[0, np.array([cos(pi/4), sin(pi/4), 0], float)]
        expectedOrigin = np.zeros(3, float)

        assert_allclose(frame.q, expectedQ, atol=1e-9)
        assert_allclose(frame.origin, expectedOrigin, atol=1e-9)

    
    def testMoveFrameLocalLocal(self):

        frame = ReferenceFrame()

        ang = pi / 4
        axis = np.array([0, 0, 1], float)
        origin = np.array([0, 0, 1], float)

        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        # move locally by 45 degrees about the x axis, and translate +1 in local x
        ang1 = pi / 4
        axis1 = np.array([1, 0, 0], float)

        frame.moveFrame(origin=np.zeros(3, float),
                        axis=axis1,
                        ang=ang1,
                        trans=np.array([1,0,0], float),
                        originFrame='local',
                        moveFrame='local')

        oExp = np.array([cos(pi/4), sin(pi/4), 1], float)
        qExp = grassmann(np.r_[cos(ang1 / 2), sin(ang1 / 2)*np.array([cos(ang), sin(ang), 0], float)], np.r_[cos(ang / 2), sin(ang / 2)*axis])

        assert_allclose(frame.q, qExp, atol=1e-9)
        assert_allclose(frame.origin, oExp, atol=1e-9)


    def testMoveFrameLocalParent(self):

        frame = ReferenceFrame()

        ang = pi / 4
        axis = np.array([0, 0, 1], float)

        frame.placeAxisAngle(axis=axis, ang=ang)

        ang1 = pi / 4
        axis1 = np.array([1, 0, 0], float)

        frame.moveFrame(origin=np.zeros(3, float),
                        axis=axis1,
                        ang=ang1,
                        trans=np.array([1,0,0], float),
                        originFrame='local',
                        moveFrame='parent')

        oExp = np.array([1, 0, 0], float)
        qExp = grassmann(np.r_[cos(ang1 / 2), sin(ang1 / 2)*axis1], np.r_[cos(ang / 2), sin(ang / 2)*axis])

        assert_allclose(frame.q, qExp, atol=1e-9)
        assert_allclose(frame.origin, oExp, atol=1e-9)


    def testMoveFrameAboutOrigin(self):

        # move the frame about the origin at 0, 0 ,0 in parent coordinates
        frame = ReferenceFrame()
        frame.origin = np.array([1, 0, 0], float)

        ang1 = pi / 4
        axis1 = np.array([0, 1, 0], float)

        frame.moveFrame(origin=np.zeros(3, float),
                        axis=axis1,
                        ang=ang1,
                        originFrame='parent',
                        moveFrame='local')
        
        qExp = np.r_[cos(ang1 / 2), sin(ang1 / 2)*np.array([0, 1, 0], float)]
        oExp = np.array([cos(pi / 4), 0, -sin(pi / 4)], float)

        assert_allclose(frame.q, qExp, atol=1e-9)
        assert_allclose(frame.origin, oExp, atol=1e-9)


    def testLocal2Parent(self):

        frame = ReferenceFrame()

        axis = np.array([1, 0, 0], float)
        ang = pi / 4
        origin = np.array([10, -2, 3], float)

        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        local = np.array([0, 1, 0], float)
        parentRot = np.array([0, cos(ang), sin(ang)], float)
        parentTrans = parentRot + origin

        assert_allclose(frame.local2parent(local, incTranslation=False), parentRot, atol=1e-9)
        assert_allclose(frame.local2parent(local, incTranslation=True), parentTrans, atol=1e-9)

    
    def testParent2Local(self):

        frame = ReferenceFrame()

        axis = np.array([1, 0, 0], float)
        ang = pi / 6
        origin = np.array([20, -1, 16], float)

        frame.placeAxisAngle(axis=axis, ang=ang, origin=origin)

        parent = np.array([0, 1, 0], float)
        locRot = np.array([0, cos(-ang), sin(-ang)], float)
        locTrans = qRotate(parent - origin, np.r_[cos(-ang / 2), sin(-ang / 2)*axis])

        assert_allclose(frame.parent2local(parent, incTranslation=False), locRot, atol=1e-9)
        assert_allclose(frame.parent2local(parent, incTranslation=True), locTrans, atol=1e-9)


def visualTest():

    frameA = ReferenceFrame()
    frameA.placeAxisAngle(axis=np.array([0,0,1], float), ang=pi/4, origin=np.zeros(3,float))

    frameB = deepcopy(frameA)
    frameB.moveFrame(origin=np.zeros(3, float), axis=np.array([0,1,0], float), ang=pi/4, trans=np.array([1,0,0], float), originFrame='local', moveFrame='local')

    drawFrames([frameA, frameB])


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
    unittest.main()