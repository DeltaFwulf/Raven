import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from math import pi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raven.rigidBody import RigidBody
from raven.referenceFrame import ReferenceFrame

class RigidBodyTests(unittest.TestCase):

    def test_transformInertiaTensor(self): # move the reference of an inertia tensor, given some MoI and reference

        rb = RigidBody()
        rb.mass = 6000
        rb.com = np.array([-1, 0, 0], float)
        rb.moi = np.zeros((3, 3), float)
        rb.moi[0, 0] = 5000
        rb.moi[1, 1] = 6500
        rb.moi[2, 2] = 2500

        root = ReferenceFrame()
        root.placeAxisAngle(axis=np.array([1, 0, 0], float), ang=pi / 2, origin=np.array([-5, 0, 0]))

        moi_exp = np.zeros((3, 3), float)
        moi_exp[0, 0] = 5000
        moi_exp[1, 1] = 218500
        moi_exp[2, 2] = 222500

        assert_allclose(rb.transformInertiaTensor(root), moi_exp, atol=1e-12)
        

    def test_transformInertiaTensor_toPrincipalAxes(self): # if the reference frame is moved to the CoM, moi should be same as in solution
        
        rb = RigidBody()
        rb.mass = 6000
        rb.com = np.array([-1, 0, 0], float)
        rb.moi = np.zeros((3, 3), float)
        rb.moi[0, 0] = 5000
        rb.moi[1, 1] = 6500
        rb.moi[2, 2] = 2500
        
        root = ReferenceFrame()
        root.placeAxisAngle(axis=np.array([1, 0, 0], float), ang=pi / 2, origin=np.array([1, 0, 0], float))

        moi_exp = np.zeros((3, 3), float)
        moi_exp[0, 0] = 5000
        moi_exp[1, 1] = 2500
        moi_exp[2, 2] = 6500

        assert_allclose(rb.transformInertiaTensor(root), moi_exp, atol=1e-12)


    def test_diffref(self): # set the initial reference some distance from the CoM
        pass


if __name__ == '__main__':
    unittest.main()