import unittest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raven.mesh import mesh2d, splitShape, triangulate, intersects



class Test_intersects(unittest.TestCase):

    def test_intersection(self): # check that only valid intersections are returned

        pts = [np.array([-1, 0.1], float), np.array([1, -0.1], float), np.array([0.1, -1], float), np.array([-0.1, 1], float), np.array([-3, 3], float)]
        set_a = [[0, 1],]
        set_b = [[2, 3], [3, 4]]

        self.assertSequenceEqual(intersects(pts=pts, set_a=set_a, set_b=set_b), [[2, 3],])

    
    def test_intersection_vertical(self):

        pts = [np.array([0, 0], float), np.array([1, 0], float), np.array([0.5, -1], float), np.array([0.5, 1], float)]
        set_a = [[0, 1],]
        set_b = [[2, 3],]

        self.assertSequenceEqual(intersects(pts=pts, set_a=set_a, set_b=set_b), [[2, 3],])


    def test_no_intersection(self):

        pts = [np.array([0, 1], float), np.array([1, 2], float), np.array([1, 1], float), np.array([2, 2], float)]
        set_a = [[0, 1],]
        set_b = [[2, 3],]

        self.assertSequenceEqual(intersects(pts=pts, set_a=set_a, set_b=set_b), [])


    def test_sharedPoint(self): # if lines share a point, they intersect
        
        pts = [np.array([0, 0], float), np.array([1, 0], float), np.array([1, 1], float)]
        set_a = [[0, 1],]
        set_b = [[1, 2],]

        self.assertSequenceEqual(intersects(pts=pts, set_a=set_a, set_b=set_b), [[1, 2],])


    def test_sameLine(self): # a line intersects with itself
        
        pts = [np.array([0, 0], float), np.array([-1, -1], float)]
        set_a = [[0, 1],]
        set_b = [[0, 1],]

        self.assertSequenceEqual(intersects(pts=pts, set_a=set_a, set_b=set_b), [[0, 1],])

    
    def test_duplicateLine(self): # each line in a counts as a unique intersection, even for duplicated lines
        
        pts = [np.array([0, -1], float), np.array([0, 1], float), np.array([-1, 0], float), np.array([1, 0], float)]
        set_a = [[0, 1], [0, 1]]
        set_b = [[2, 3],]

        self.assertSequenceEqual(intersects(pts=pts, set_a=set_a, set_b=set_b), [[2, 3], [2, 3]])



class Test_triangulate(unittest.TestCase):

    def test_nominal(self): # triangulate a square

        pts = [np.array([0, 0], float), np.array([1, 0], float), np.array([1, 1], float), np.array([0, 1], float)]
        
        expected = [(0, 1, 2), (0, 2, 3)]

        self.assertSequenceEqual(triangulate(pts), expected)



class Test_split(unittest.TestCase):

    def test_nosplit(self):
        pts = [np.array([1, 5], float), np.array([0, 3], float), np.array([0, 1], float), np.array([1, 0], float), np.array([2, 2], float), np.array([2, 4], float)]
        c = [0, 1, 2, 3, 4, 5]

        out = splitShape(pts_global=pts, c=c)

        self.assertEqual(len(out), 1)
        self.assertSequenceEqual(out, [c,])


    def test_inflection(self):
        pts = [np.array([1, 4], float), np.array([0, 2], float), np.array([0.5, 2.5], float), np.array([1, 0], float), np.array([2, 3], float)]
        c = [0, 1, 2, 3, 4]

        out = splitShape(pts_global=pts, c=c)

        self.assertEqual(len(out), 3)
        self.assertSequenceEqual(out, [[0, 2, 4], [0, 1, 2], [2, 3, 4]])


    def test_horizontal(self): # horizontal lines should not incur a split
        
        pts = [np.array([0, 0], float), np.array([1, 0], float), np.array([1, 1], float), np.array([0, 1], float)]
        c = [0, 1, 2, 3]

        out = splitShape(pts_global=pts, c=c)

        self.assertEqual(len(out), 1)
        self.assertSequenceEqual(out, [c,])



class Test_mesh2d(unittest.TestCase):

    def meshSquare(self):
        
        pts = [np.array([0, 0], float), np.array([])]



if __name__ == '__main__':
    unittest.main()