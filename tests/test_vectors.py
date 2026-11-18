"""Unit test cases for vector functions."""

from math import pi
import unittest
import numpy as np

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raven.vectors import cartesian2coords, coords2cartesian, ntransform


class TestCoordinateTransforms(unittest.TestCase):
    """Tests transformations between coordinates and cartesian vectors."""

    def test_cartesian2coords(self):
        """parity checks"""

        # Northeast a
        np.testing.assert_allclose(cartesian2coords(np.array([100.0, 100.0, 100.0])),
                                   np.array([0.785398163397448310, 0.615479708670387343, 173.205080756887729]))

        # Northeast b
        np.testing.assert_allclose(cartesian2coords(np.array([-100.0, 100.0, 100.0])),
                                   np.array([2.35619449019234493, 0.615479708670387343, 173.205080756887729]))

        # Northwest a
        np.testing.assert_allclose(cartesian2coords(np.array([-100.0, -100.0, 100.0])),
                                   np.array([3.92699081698724155, 0.615479708670387343, 173.205080756887729]))

        # Northwest b
        np.testing.assert_allclose(cartesian2coords(np.array([100.0, -100.0, 100.0])),
                                   np.array([5.49778714378213817, 0.615479708670387343, 173.205080756887729]))

        # Southeast a
        np.testing.assert_allclose(cartesian2coords(np.array([100.0, 100.0, -100.0])),
                                   np.array([0.785398163397448310, -0.615479708670387343, 173.205080756887729]))

        # Southeast b
        np.testing.assert_allclose(cartesian2coords(np.array([-100.0, 100.0, -100.0])),
                                   np.array([2.35619449019234493, -0.615479708670387343, 173.205080756887729]))

        # Southwest a
        np.testing.assert_allclose(cartesian2coords(np.array([-100.0, -100.0, -100.0])),
                                   np.array([3.92699081698724155, -0.615479708670387343, 173.205080756887729]))

        # Southwest b
        np.testing.assert_allclose(cartesian2coords(np.array([100.0, -100.0, -100.0])),
                                   np.array([5.49778714378213817, -0.615479708670387343, 173.205080756887729]))


    def test_coords2cartesian(self):
        """convert lat, long, r to xyz"""

        # Northeast
        expected = np.array([0.612372435695794525, 0.612372435695794525, 0.5])
        np.testing.assert_allclose(coords2cartesian(lat=pi / 6, lon=pi / 4, r=1.0), expected)

        # Northwest
        expected = np.array([-0.654508497187473712, -0.475528258147576786, 0.587785252292473129])
        np.testing.assert_allclose(coords2cartesian(lat= pi / 5, lon=1.2*pi, r=1.0), expected)

        # Southeast
        expected = np.array([0.654508497187473712, 0.475528258147576786, -0.587785252292473129])
        np.testing.assert_allclose(coords2cartesian(lat=-pi / 5, lon=0.2*pi, r=1.0), expected)

        # Southwest
        expected = np.array([0.572061402817684298, -0.415626937777453429, -0.707106781186547524])
        np.testing.assert_allclose(coords2cartesian(lat=-pi/4, lon=1.8*pi, r=1.0), expected)


    def test_coords2cartesian_latbounds(self):
        """latitude between +- 90"""

        with self.assertRaises(ValueError):
            coords2cartesian(lat=1.1*pi, lon=0.0, r=1.0)


    def test_coords2cartesian_lonbounds(self):
        """longitude between 0 and 2*pi"""

        with self.assertRaises(ValueError):
            coords2cartesian(lat=0.0, lon=2.1*pi, r=1.0)


    def test_coords2cartesian_rbounds(self):
        """Radius must be >= 0"""

        with self.assertRaises(ValueError):
            coords2cartesian(lat=0.0, lon=0.0, r=-1.0)


class TestNTransform(unittest.TestCase):
    """Test the ntransform function."""

    def test_n_by_3(self):
        """Should return correct nx3 vector."""

        vecs = np.array([[1, 1, 1], [-1, -1, -1]])
        q = np.array([[0.923879532511286756, 0, 0.382683432365089772, 0], [0.965925826289068287, 0.258819045102520762, 0, 0]])
        origin = np.array([[2, 2, 2], [3, 3, 3]])
        expected = np.array([[3.41421356237309505, 3, 2], [2, 2.63397459621556135, 1.63397459621556135]])
        np.testing.assert_allclose(ntransform(vecs, q, origin), expected)


    def test_invert(self):
        """Invert flag should invert transformation."""

        vecs = np.array([[3.41421356237309505, 3, 2], [2, 2.63397459621556135, 1.63397459621556135]])
        q = np.array([[0.923879532511286756, 0, 0.382683432365089772, 0], [0.965925826289068287, 0.258819045102520762, 0, 0]])
        origin = np.array([[2, 2, 2], [3, 3, 3]])
        expected = np.array([[1, 1, 1], [-1, -1, -1]])

        np.testing.assert_allclose(ntransform(vecs, q, origin, invert=True), expected)


    def test_single(self):
        """Should successfully transform a 1d array."""

        vec = np.array([1, 0, 0])
        q = np.array([0.707106781186547524, 0.0, 0.0, 0.707106781186547524])
        origin = np.zeros(3)
        expected = np.array([0, 1, 0])

        np.testing.assert_allclose(ntransform(vec, q, origin)[0], expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
