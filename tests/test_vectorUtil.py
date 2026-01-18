import unittest
import numpy as np
from math import pi, sin, cos, sqrt
from numpy.testing import assert_allclose
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorUtil import cartesian2spherical, spherical2cartesian, cartesian2coords, coords2cartesian, spherical2coords, coords2spherical, getAngleSigned, getAngleUnsigned, projectVector, grassmann, qRotate, unit



class test_vector_util(unittest.TestCase):

    # coordinate transformations
    def test_cart2sphere(self):

        incExp = pi / 6
        azExp = pi / 6
        rExp = 1.0

        vector = np.array([rExp*cos(azExp)*sin(incExp), rExp*sin(azExp)*sin(incExp), rExp*cos(incExp)], float)

        r, inc, az = cartesian2spherical(vector)

        self.assertAlmostEqual(r, rExp, places=9)
        self.assertAlmostEqual(inc, incExp ,places=9)
        self.assertAlmostEqual(az, azExp, places=9)
        
        
    def test_cart2sphere_nullangs(self):

        incExp = 0
        azExp = 0
        rExp = 1

        vec = rExp*np.array([cos(azExp)*sin(incExp), sin(azExp)*sin(incExp), cos(incExp)])

        r, inc, az = cartesian2spherical(vec)

        self.assertAlmostEqual(r, rExp, places=9)
        self.assertAlmostEqual(inc, incExp ,places=9)
        self.assertAlmostEqual(az, azExp, places=9)


    def test_cart2sphere_nullvec(self):

        vec = np.zeros(3, float)
        r, inc, az = cartesian2spherical(vec)

        self.assertEqual(r, 0)
        self.assertEqual(inc, 0)
        self.assertEqual(az, 0)


    def test_sphere2cart(self):

        inc = pi / 4
        az = pi / 6
        r = 1.0

        x = sin(inc)*cos(az)
        y = sin(inc)*sin(az)
        z = cos(inc)
        vecExp = r*np.r_[x, y, z]
        assert_allclose(spherical2cartesian(inc, az, r), vecExp, atol=1e-9)

    
    def test_sphere2cart_biginc(self):
        
        inc = 5*pi / 4
        az = pi / 4
        r = 1
        vecExp = np.r_[sin(3*pi / 4)*cos(5*pi / 4), sin(3*pi / 4)*sin(5*pi / 4), cos(3*pi / 4)]
        assert_allclose(spherical2cartesian(inc, az, r), vecExp, atol=1e-9)


    def test_sphere2cart_zeroradius(self):
        assert_allclose(spherical2cartesian(inc=0, az=0, r=0), np.zeros(3, float), atol=1e-9)

    
    def test_cartesian2coords(self):

        lat, long = cartesian2coords(np.r_[sqrt(3/8), -sqrt(3/8), 0.5])

        self.assertAlmostEqual(lat, pi / 6 ,places=9)
        self.assertAlmostEqual(long, -pi / 4, places=9)

    
    def test_cartesian2coords_nullvec(self):

        lat, long = cartesian2coords(np.zeros(3, float))

        self.assertAlmostEqual(lat, pi / 2, places=9)
        self.assertAlmostEqual(long, 0, places=9)


    def test_coords2cartesian(self):
        pass


    def test_sphere2coord(self):

        lat, long = spherical2coords(pi / 6, pi / 10)

        self.assertAlmostEqual(lat, pi / 3, places=9)
        self.assertAlmostEqual(long, pi / 10, places=9)


    def test_sphere2coord_biginc(self):

        inc = 5*pi / 4
        az = pi / 6

        latExp = -pi / 4
        longExp = -5*pi / 6

        lat, long = spherical2coords(inc, az)

        self.assertAlmostEqual(lat, latExp, places=9)
        self.assertAlmostEqual(long, longExp, places=9)

    
    def test_sphere2coord_bigaz(self): # Tests if conversion maintains east positive convention

        inc = pi / 6
        az = 6*pi / 4

        latExp = pi / 3
        longExp = -pi / 2

        lat, long = spherical2coords(inc, az)

        self.assertAlmostEqual(lat, latExp, places=9)
        self.assertAlmostEqual(long, longExp, places=9)


    def test_coord2sphere(self):

        lat = pi / 6
        long = pi / 3

        incExp = pi / 3
        azExp = pi / 3

        inc, az = coords2spherical(lat, long)

        self.assertAlmostEqual(inc, incExp, places=9)
        self.assertAlmostEqual(az, azExp, places=9)


    def test_coord2sphere_biglat(self):

        lat = 5*pi / 4 # 1.25 * pi = 225 degrees north, 45 degrees south
        long = pi / 3

        incExp = 3*pi / 4 # 45 degrees south
        azExp = 4*pi / 3

        inc, az = coords2spherical(lat, long)

        self.assertAlmostEqual(inc, incExp, places=9)
        self.assertAlmostEqual(az, azExp, places=9)


    def test_coords2sphere_biglong(self):

        lat = pi / 6
        long = 9*pi / 4

        incExp = pi / 3
        azExp = pi / 4

        inc, az = coords2spherical(lat, long)
        
        self.assertAlmostEqual(inc, incExp, places=9)
        self.assertAlmostEqual(az, azExp, places=9)


    def test_angleSigned(self):

        vecA = np.array([1, 0, 0], float)
        vecB = np.array([0, 1, 0], float)
        normal = np.array([0, 0, 1], float)

        self.assertEqual(getAngleSigned(vecA, vecB, normal), pi / 2)
        self.assertEqual(getAngleSigned(vecB, vecA, normal), -pi / 2)
        self.assertEqual(getAngleSigned(vecA, vecB, -normal), -pi / 2)


    def test_angleSigned_notnormal(self):
        
        vecA = np.array([1, 0, 0], float)
        vecB = np.array([0, 1, 0], float)
        badnormal = np.array([1, 1, 0], float)

        with self.assertRaises(ValueError):
            getAngleSigned(vecA, vecB, badnormal)


    def test_angleSigned_parallel(self):

        vec = np.array([1, 0, 0], float)
        normal = np.array([0, 1 ,0], float)

        self.assertEqual(getAngleSigned(vec, vec, normal), 0)

    
    def test_angleSigned_zerolen(self):

        vecA = np.array([1, 0, 0], float)
        vecB = np.zeros(3, float)
        normal = np.array([0, 0, 1], float)

        with self.assertRaises(ValueError):
            getAngleSigned(vecA, vecB, normal)


    def test_angleUnsigned(self):

        vecA = np.array([1, 0, 0], float)
        vecB = np.array([0, 0, 1], float)

        self.assertEqual(getAngleUnsigned(vecA, vecB), pi / 2)
        self.assertEqual(getAngleUnsigned(vecB, vecA), pi / 2)


    def test_angleUnsigned_parallel(self):

        vec = np.array([1, 1, 1], float)
        self.assertEqual(getAngleUnsigned(vec, vec), 0)

    
    def test_angleUnsigned_zerolen(self):

        vecA = np.array([1, 0, 0], float)
        vecB = np.zeros(3, float)

        with self.assertRaises(ValueError):
            getAngleUnsigned(vecA, vecB)


    def test_projectVector(self):

        vecA = np.array([2.1, 0.8, -3.0], float)
        vecB = np.array([1, 1, 0], float)
        
        assert_allclose(projectVector(vecA, vecB, normal=False), np.array([1.45, 1.45, 0], float), atol=1e-9)
        assert_allclose(projectVector(vecA, vecB, normal=True), np.array([0.65, -0.65, -3.0]), atol=1e-9)


    def test_projectVector_nulla(self):

        vecA = np.zeros(3, float)
        vecB = np.array([1, 1, 0], float)

        assert_allclose(projectVector(vecA, vecB, normal=False), np.zeros(3, float), atol=1e-9)
        assert_allclose(projectVector(vecA, vecB, normal=True), np.zeros(3, float), atol=1e-9)


    def test_projectVector_nullb(self):

        vecA = np.array([2.1, 0.8, -3.0], float)
        vecB = np.zeros(3, float)

        with self.assertRaises(ValueError):
            projectVector(vecA, vecB, normal=False)


    def test_projectVector_badnormal(self):

        vecA = np.array([1, 0, 0], float)
        vecB = np.array([1, 2, 3], float)

        with self.assertRaises(ValueError):
            projectVector(vecA, vecB, normal='normal')


    # grassmann
    def test_grassmann(self):

        axis = np.array([1, 0, 0])
        halfAng = pi / 8

        qa = np.r_[cos(halfAng / 2), sin(halfAng / 2)*axis]
        qb = np.r_[cos(halfAng / 2), sin(halfAng / 2)*axis]

        qExp = np.r_[cos(halfAng), sin(halfAng)*axis]

        assert_allclose(grassmann(qa, qb), qExp, atol=1e-9)

    
    def test_qRotate(self):

        vec = np.array([1, 1, 1], float)
        q = np.r_[cos(pi / 2), sin(pi / 2)*np.array([1, 0, 0], float)]
        
        assert_allclose(qRotate(vec, q), np.array([1, -1, -1], float), atol=1e-9)


    def test_qRotate_nullvec(self):

        vec = np.zeros(3, float)
        q = np.r_[cos(pi / 2), sin(pi / 2)*np.array([1, 0, 0], float)]

        assert_allclose(qRotate(vec, q), np.zeros(3, float), atol=1e-9)


    def test_unit(self):

        vec = np.array([-1, 2, 2], float)
        assert_allclose(unit(vec), (1 / 3)*vec, atol=1e-9)


    def test_unit_nullvec(self):

        with self.assertRaises(ValueError):
            unit(np.zeros(3, float))



if __name__ == '__main__':
    unittest.main()