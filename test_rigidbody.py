import unittest
import numpy as np
from numpy.testing import assert_allclose
from math import pi, sqrt

from primitives import Conic, RectangularPrism, TriangularPrism
from referenceFrame import ReferenceFrame



class ConicTests(unittest.TestCase):

    def testInit(self):

        density = 1000.0
        length = 1.0
        dOuterRoot = 1.0
        dOuterEnd = 1.0
        dInnerRoot = 0.5
        dInnerEnd = 0.5

        conic = Conic(density=density,
                      length=length,
                      dOuterRoot=dOuterRoot,
                      dOuterEnd=dOuterEnd,
                      dInnerRoot=dInnerRoot,
                      dInnerEnd=dInnerEnd)
        
        self.assertEqual(conic.length, 1.0)
        self.assertEqual(conic.rOuterRoot, 0.5)
        self.assertEqual(conic.rOuterEnd, 0.5)
        self.assertEqual(conic.rInnerRoot, 0.25)
        self.assertEqual(conic.rInnerEnd, 0.25)
        
        self.assertIsInstance(conic.name, str)
        self.assertIsInstance(conic.mass, float)
        self.assertIsInstance(conic.com, np.ndarray)
        self.assertIsInstance(conic.moi, np.ndarray)
        self.assertIsInstance(conic.pts, np.ndarray)
        self.assertIsInstance(conic.tris, list)


    def test_init_noinner(self):

        conic = Conic(density = 1.0,
                      length = 1.0,
                      dOuterRoot=1.0,
                      dOuterEnd = 1.0)
        
        self.assertEqual(conic.length, 1.0)
        self.assertEqual(conic.rOuterRoot, 0.5)
        self.assertEqual(conic.rOuterEnd, 0.5)
        self.assertEqual(conic.rInnerRoot, 0)
        self.assertEqual(conic.rInnerEnd, 0)

        self.assertIsInstance(conic.mass, float)
        self.assertIsInstance(conic.com, np.ndarray)
        self.assertIsInstance(conic.moi, np.ndarray)

        self.assertIsInstance(conic.pts, np.ndarray)
        self.assertIsInstance(conic.tris, list)

    
    def test_init_zerolen(self):

        with self.assertRaises(ValueError):
            Conic(density = 1.0,
                  length = 0,
                  dOuterRoot=1.0,
                  dOuterEnd = 1.0)
        
    
    def test_init_neglen(self):

        with self.assertRaises(ValueError):
            Conic(density=1.0,
                  length=-1,
                  dOuterRoot=1.0,
                  dOuterEnd=1.0)
            
    
    def test_init_nulldiameter(self):

        with self.assertRaises(ValueError):
            Conic(density=1.0,
                  length=1.0,
                  dOuterRoot=0,
                  dOuterEnd=0)
        

    def test_biginner(self):

        with self.assertRaises(ValueError):
            Conic(density=1.0,
                  length=1.0,
                  dOuterRoot=1.0,
                  dOuterEnd=1.0,
                  dInnerRoot=2.0)
            
        with self.assertRaises(ValueError):
            Conic(density=1.0,
                  length=1.0,
                  dOuterRoot=1.0,
                  dOuterEnd=1.0,
                  dInnerEnd=2.0)


    def test_neg_inner(self):

        with self.assertRaises(ValueError):
            Conic(density=1.0,
                  length=1.0,
                  dOuterRoot=1.0,
                  dOuterEnd=1.0,
                  dInnerRoot=-1.0)
        
        with self.assertRaises(ValueError):
            Conic(density=1.0,
                  length=1.0,
                  dOuterRoot=1.0,
                  dOuterEnd=1.0,
                  dInnerEnd=-1.0)
            

    def test_solidcylinder(self):
        
        cyl = Conic(density=1000,
                     length=2,
                     dOuterRoot=1,
                     dOuterEnd=1)
        
        self.assertAlmostEqual(cyl.mass, 500*pi, places=12)
        assert_allclose(cyl.com, np .array([-1, 0, 0], float), rtol=1e-9)
        moi_exp = 125 / 12 *pi*np.array([[6, 0, 0], [0, 19, 0], [0, 0, 19]], float)
        assert_allclose(cyl.moi, moi_exp, atol=1e-9)


    def test_solidfrustum(self):

        density = 1000
        length = 0.8
        dSmall = 1
        dBig = 4

        mf = 1400*pi
        moi = np.zeros((3, 3), float)
        moi[0,0] = 1705*pi
        moi[1,1] = 190049*pi / 210
        moi[2,2] = moi[1,1]

        frustum = Conic(density=density,
                        length=length,
                        dOuterRoot=dSmall,
                        dOuterEnd=dBig)
        
        self.assertAlmostEqual(frustum.mass, mf, places=12)
        assert_allclose(frustum.com, np.array([-19 / 35, 0, 0], float), atol=1e-12)
        assert_allclose(frustum.moi, moi, atol=1e-12)

        # parity B check
        frustum = Conic(density=density,
                        length=length,
                        dOuterRoot=dBig,
                        dOuterEnd=dSmall)
        
        self.assertAlmostEqual(frustum.mass, mf, places=12)
        assert_allclose(frustum.com, np.array([-9 / 35, 0, 0]), atol=1e-12)
        assert_allclose(frustum.moi, moi, atol=1e-12)


    def test_solidcone(self):
        
        cone = Conic(density=1000,
                     length=1.5,
                     dOuterRoot=0,
                     dOuterEnd=4)

        moi = np.zeros((3, 3), float)
        moi[0, 0] = 2400*pi
        moi[1, 1] = 1368.75*pi
        moi[2, 2] = moi[1, 1]

        self.assertAlmostEqual(cone.mass, 2000*pi, places=12)
        assert_allclose(cone.com, np.array([-9 / 8, 0, 0]), atol=1e-12)
        assert_allclose(cone.moi, moi, atol=1e-12)

        # parity B check
        cone = Conic(density=1000,
                     length=1.5,
                     dOuterRoot=4,
                     dOuterEnd=0)
        
        self.assertAlmostEqual(cone.mass, 2000*pi, places=12)
        assert_allclose(cone.com, np.array([-3 / 8, 0, 0], float), atol=1e-12)
        assert_allclose(cone.moi, moi, atol=1e-12)


    def test_hollow(self):
        
        hollow = Conic(density=1000,
                       length=0.8,
                       dOuterRoot=1,
                       dOuterEnd=4,
                       dInnerRoot=0.5,
                       dInnerEnd=0)
        
        moi = np.zeros((3, 3), float)
        moi[0, 0] = pi*13582950 / 7968
        moi[1, 1] = pi*7190771 / 7968
        moi[2, 2] = moi[1, 1]
        
        self.assertAlmostEqual(hollow.mass, 4150*pi / 3, places=11)
        assert_allclose(hollow.com, np.array([-227 / 415, 0, 0], float), atol=1e-12)
        assert_allclose(hollow.moi, moi, atol=1e-12)



class RectPrismTests(unittest.TestCase):

    def test_init(self):

        box = RectangularPrism(density=1000, x=1, y=1, z=1)

        self.assertIsInstance(box.name, str)
        self.assertIsInstance(box.mass, float)
        self.assertIsInstance(box.com, np.ndarray)
        self.assertIsInstance(box.moi, np.ndarray)
        self.assertIsInstance(box.pts, np.ndarray)
        self.assertIsInstance(box.tris, list)


    def test_nominalinput(self):

        rect = RectangularPrism(density=1000,
                               x=0.2,
                               y=0.5,
                               z=2)
        
        moi = np.zeros((3, 3), float)

        moi[0, 0] = 425 / 6
        moi[1, 1] = 202 / 3
        moi[2, 2] = 29 / 6
        
        self.assertEqual(rect.mass, 200)
        assert_allclose(rect.com, np.array([-0.1, 0, 0], float), atol=1e-12)
        assert_allclose(rect.moi, moi, atol=1e-12)


    def test_nullinputs(self):

        with self.assertRaises(ValueError):
            RectangularPrism(density=0,
                             x=1,
                             y=1,
                             z=1)
        
        with self.assertRaises(ValueError):
            RectangularPrism(density=1,
                             x=0,
                             y=1,
                             z=1)
            
        with self.assertRaises(ValueError):
            RectangularPrism(density=1,
                             x=1,
                             y=0,
                             z=1)
            
        with self.assertRaises(ValueError):
            RectangularPrism(density=1,
                             x=1,
                             y=1,
                             z=0)
        

    def test_neginputs(self):

        with self.assertRaises(ValueError):
            RectangularPrism(density=-1,
                             x=1,
                             y=1,
                             z=1)
        
        with self.assertRaises(ValueError):
            RectangularPrism(density=1,
                             x=-1,
                             y=1,
                             z=1)
            
        with self.assertRaises(ValueError):
            RectangularPrism(density=1,
                             x=1,
                             y=-1,
                             z=1)
            
        with self.assertRaises(ValueError):
            RectangularPrism(density=1,
                             x=1,
                             y=1,
                             z=-1)
            


class TriangularPrismTests(unittest.TestCase):

    def test_build(self):

        pts = [np.zeros(2, float), np.array([0.5, sqrt(3) / 2], float), np.array([1, 0])]
        tri = TriangularPrism(density=1000, thickness=1, pts=pts)

        self.assertIsInstance(tri.name, str)
        self.assertIsInstance(tri.mass, float)
        self.assertIsInstance(tri.com, np.ndarray)
        self.assertIsInstance(tri.moi, np.ndarray)

        
    def test_positiveCase(self):

        pts = [np.zeros(2, float), np.array([0.5, sqrt(3) / 2], float), np.array([1, 0])]
        tri = TriangularPrism(density=1000, thickness=1, pts=pts)

        self.assertAlmostEqual(tri.mass, 1000*sqrt(3) / 4, 9)
        assert_allclose(tri.com, np.array([-0.5, 0.5, sqrt(3) / 6], float), atol=1e-9)

        I = np.zeros((3, 3), float)
        I[0, 0] = 2
        I[1, 1] = 3
        I[2, 2] = 3
        I *= 125 / (4*sqrt(3))

        assert_allclose(tri.moi, I, atol=1e-9)
        
    
    def test_negativeCase(self):
        
        pts = [np.zeros(2, float), np.array([0.5, -sqrt(3) / 2], float), np.array([1, 0])]
        tri = TriangularPrism(density=1000, thickness=1, pts=pts)

        self.assertAlmostEqual(tri.mass, 1000*sqrt(3) / 4, 9)
        assert_allclose(tri.com, np.array([-0.5, 0.5, -sqrt(3) / 6], float), atol=1e-9)

        I = np.zeros((3, 3), float)
        I[0, 0] = 2
        I[1, 1] = 3
        I[2, 2] = 3
        I *= 125 / (4*sqrt(3))

        assert_allclose(tri.moi, I, atol=1e-9)



    def test_rootAtLineOfAction(self): # points do not have to have a 'root' point
        pass


    def test_nullEdge(self):

        pts = [np.array([0, 0], float), np.array([0.5, sqrt(3) / 2], float), np.array([0, 0], float)]
        
        with self.assertRaises(ValueError):
            TriangularPrism(density=1000, thickness=1, pts=pts)

    
    def test_zeroThickness(self):
        
        pts = [np.zeros(2, float), np.array([0.5, sqrt(3) / 2], float), np.array([1, 0])]
        
        with self.assertRaises(ValueError):
            TriangularPrism(density=1000, thickness=0, pts=pts)

    
    def test_negativeThickness(self):
        
        pts = [np.zeros(2, float), np.array([0.5, sqrt(3) / 2], float), np.array([1, 0])]
        
        with self.assertRaises(ValueError):
            TriangularPrism(density=1000, thickness=-1, pts=pts)


    def test_zeroDensity(self):
        
        pts = [np.zeros(2, float), np.array([0.5, sqrt(3) / 2], float), np.array([1, 0])]
        
        with self.assertRaises(ValueError):
            TriangularPrism(density=0, thickness=1, pts=pts)


    def test_negativeDensity(self):
        
        pts = [np.zeros(2, float), np.array([0.5, sqrt(3) / 2], float), np.array([1, 0])]
        
        with self.assertRaises(ValueError):
            TriangularPrism(density=-1000, thickness=1, pts=pts)



class RigidBodyTests(unittest.TestCase):

    def test_moveCuboid(self):

        rect = RectangularPrism(density=1000,
                                x=2,
                                y=1,
                                z=3)
        
        root = ReferenceFrame()
        root.placeAxisAngle(axis=np.array([1, 0, 0], float), ang=pi / 2, origin=np.array([-5, 0, 0]))

        moi_exp = np.zeros((3, 3), float)
        moi_exp[0, 0] = 5000
        moi_exp[1, 1] = 218500
        moi_exp[2, 2] = 222500

        assert_allclose(rect.transformInertiaTensor(root), moi_exp, atol=1e-12)
        

    def test_movetocom(self):
        
        rect = RectangularPrism(density=1000,
                                x=2,
                                y=1,
                                z=3)
        
        root = ReferenceFrame()
        root.placeAxisAngle(axis=np.array([1, 0, 0], float), ang=pi / 2, origin=np.array([1, 0, 0], float))

        moi_exp = np.zeros((3, 3), float)
        moi_exp[0, 0] = 5000
        moi_exp[1, 1] = 2500
        moi_exp[2, 2] = 6500

        assert_allclose(rect.transformInertiaTensor(root), moi_exp, atol=1e-12)


    def test_diffref(self): # this will be done with the TriangularPrism primitive, as I calculated from root not com?
        pass


if __name__ == '__main__':
    unittest.main()