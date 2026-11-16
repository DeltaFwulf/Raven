"""Validate the COESA76 model."""

import unittest
# import sys
# import os
import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raven.atmosphere import coesa_low, coesa_high, coesa76



class TestCoesaLow(unittest.TestCase):
    """Tests for the low coesa model (<86km)"""

    def test_troposphere(self):
        """Troposphere, h = 0-11 km."""

        expected = np.array([255.675543221803506, 54048.2809123108792, 0.736428534307431382])
        np.testing.assert_allclose(coesa_low(5.0), expected, atol=1e-9)

    def test_stratosphere_low(self):
        """Low stratosphere, h = 11-20 km."""

        expected = np.array([216.65, 12111.8224049626142, 0.194755042372752681])
        np.testing.assert_allclose(coesa_low(15.0), expected, atol=1e-9)


    def test_stratosphere_mid(self):
        """Mid stratosphere, h = 20-32 km."""

        expected = np.array([222.544089803699525, 2188.37728893941023, 0.0342565832217336284])
        np.testing.assert_allclose(coesa_low(26.0), expected, atol=1e-9)


    def test_stratosphere_high(self):
        """High stratosphere, h = 32-47 km."""

        expected = np.array([250.349646102421130, 287.143911565033126, 0.00399567853249466275])
        np.testing.assert_allclose(coesa_low(40.0), expected, atol=1e-9)


    def test_mesosphere_low(self):
        """Low mesosphere, h = 47-51 km."""

        expected = np.array([270.65, 90.3367856710788326, 0.00116277185776535789])
        np.testing.assert_allclose(coesa_low(49.0), expected, atol=1e-9)


    def test_mesosphere_mid(self):
        """Mid mesosphere, h = 51-71 km."""

        expected = np.array([244.273430957127449, 19.1575308953544718, 0.000273212906979401250])
        np.testing.assert_allclose(coesa_low(61.0), expected, atol=1e-9)


    def test_mesosphere_high(self):
        """High mesosphere, h = 71-84.852 km"""

        expected = np.array([202.540977853740136, 1.46736705231354269, 0.0000252385134250865798])
        np.testing.assert_allclose(coesa_low(78.0), expected, atol=1e-9)



class TestCoesaHigh(unittest.TestCase):
    """Covers high altitudes, 86 to 1000 km."""

    def test_band0(self):
        """z = 86 - 91 km"""

        expected = np.array([186.8673, 0.261731938076420245, 0.00000487490453834568150])
        np.testing.assert_allclose(coesa_high(88.0), expected, atol=1e-9)


    def test_band1(self):
        """z = 91-100 km"""

        expected = np.array([188.418276403113245, 0.0759611127680710342, 0.00000139352046198986716])
        np.testing.assert_allclose(coesa_high(95.0), expected, atol=1e-9)


    def test_band2(self):
        """z = 100-110 km"""

        expected = np.array([208.835194282161851, 0.0144745094943199799, 0.000000232441684554455030])
        np.testing.assert_allclose(coesa_high(105.0), expected, atol=1e-9)


    def test_band3(self):
        """z = 110-120 km"""

        expected = np.array([300.0, 0.00400883655769173230, 0.0000000428834341901961556])
        np.testing.assert_allclose(coesa_high(115.0), expected, atol=1e-9, rtol=1e-6)


    def test_band4(self):
        """z = 120-150 km"""

        expected = np.array([516.588606238588412, 0.000935821056819769153, 0.00000000544266082012721465])
        np.testing.assert_allclose(coesa_high(135.0), expected, atol=1e-9, rtol=1e-6)


    def test_band5(self):
        """z = 150-200 km"""

        expected = np.array([769.810772060024099, 0.000179363539068311872, 0.000000000633843860427942691])
        np.testing.assert_allclose(coesa_high(175.0), expected, atol=1e-9, rtol=1e-6)

    def test_band6(self):
        """z = 200-300 km"""

        expected = np.array([941.3298915463327670, 0.0000247670874226870982, 0.0000000000607254602086042996])
        np.testing.assert_allclose(coesa_high(250.0), expected, atol=1e-9, rtol=1e-6)


    def test_band7(self):
        """z = 300-500 km"""

        expected = np.array([995.825361036596612, 0.00000145178887358633095, 0.00000000000280273172017932380])
        np.testing.assert_allclose(coesa_high(400.0), expected, atol=1e-9, rtol=1e-6)

    def test_band8(self):
        """z = 500-750 km"""

        expected = np.array([999.901961509089913, 0.0000000626044279910795346, 0.0000000000000799748561578211642])
        np.testing.assert_allclose(coesa_high(625.0), expected, atol=1e-9, rtol=1e-6)


    def test_band9(self):
        """z = 750-1000 km"""

        expected = np.array([999.998004503163253, 0.0000000120421143354973974, 0.00000000000000666410090476932300])
        np.testing.assert_allclose(coesa_high(875.0), expected, atol=1e-9, rtol=1e-6)


class TestCoesa76(unittest.TestCase):
    """Tests the top-level coesa76 function."""

    def test_bounds(self):
        """If not within 0 or 1000 km, return all zeros."""

        np.testing.assert_equal(np.zeros(3), coesa76(-1))
        np.testing.assert_equal(np.zeros(3), coesa76(1.1e6))

    def test_high(self):
        """Values between 86000 and 1e6 should call the coesa_high subroutine."""

        expected = np.array([995.825361036596612, 0.00000145178887358633095, 0.00000000000280273172017932380])
        np.testing.assert_allclose(coesa76(400000.0), expected, atol=1e-9, rtol=1e-6)

    def test_low(self):
        """Values from 0 to 86000 should call the coesa_low subroutine."""

        expected = np.array([270.65, 90.3367856710788326, 0.00116277185776535789])
        np.testing.assert_allclose(coesa76(49000.0), expected, atol=1e-9)

if __name__ == '__main__':
    unittest.main()
