import unittest
from numpy.testing import assert_array_almost_equal
from hypertiling import HyperbolicTiling

# note: central cell not identical due to angular or translational offsets used by some kernels


class TestTilingIterator(unittest.TestCase):


    def test_iterator_SRG(self):
        expected_output = [
            -1.00000000e-09+0.j,
            5.91534875e-01+0.0012746j,
            -1.27459976e-03+0.59153488j,
            -5.91534876e-01-0.0012746j,
            1.27459706e-03-0.59153488j
        ]

        T = HyperbolicTiling(4, 7, 1, kernel="SRG")
        for poly in T:
            assert_array_almost_equal(poly, expected_output, decimal=7)



    def test_iterator_SRS(self):
        expected_output = [
            0.0        +0.j, 
            0.59153488 +0.0012746j, 
            -0.0012746 +0.59153488j,
            -0.59153488-0.0012746j, 
            0.0012746  -0.59153488j
        ]

        T = HyperbolicTiling(4, 7, 1, kernel="SRS")
        for poly in T:
            assert_array_almost_equal(poly, expected_output, decimal=7)



    def test_iterator_DUN07X(self):
        expected_output = [
            0.00000000e+00 + 0.00000000e+00j,  
            5.91536249e-01 + 0.00000000e+00j,
            3.62211487e-17 + 5.91536249e-01j, 
            -5.91536249e-01 + 7.24422974e-17j,
            -1.08663446e-16 - 5.91536249e-01j
        ]

        T = HyperbolicTiling(4, 7, 1, kernel="DUN07X")
        for poly in T:
            assert_array_almost_equal(poly, expected_output, decimal=7)



#    def test_iterator_GR(self):
#        expected_output = [
#            0.        +0.j,
#            0.59035233+0.03740675j,
#            -0.03740675+0.59035233j,
#            -0.59035233-0.03740675j,
#            0.03740675-0.59035233j
#        ]
#
#        T = HyperbolicTiling(4, 7, 1, kernel="GR")
#        for poly in T:
#            assert_array_almost_equal(poly, expected_output, decimal=7)