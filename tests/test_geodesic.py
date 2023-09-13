import unittest
from hypertiling.geodesics import *

class TestGeodesic(unittest.TestCase):

    def test_unit_circle_inverson(self):
        self.assertEqual(unit_circle_inversion(1), 1)
        self.assertEqual(unit_circle_inversion(1j), 1j)
        self.assertEqual(unit_circle_inversion(0.5), 2)
        self.assertEqual(unit_circle_inversion(-0.5j), -2j)


    def test_circle_through_three_points(self):
        self.assertEqual(circle_through_three_points(1, 1j, -1), (0, 1))
        self.assertEqual(circle_through_three_points(1, 1j, -1j), (0, 1))
        self.assertEqual(circle_through_three_points(1, -2j, -1), (- 0.75j, 1.25))

    def test_geodesic_midpoint(self):
        self.assertAlmostEqual(geodesic_midpoint(0.5, -0.5), 0)
        self.assertAlmostEqual(geodesic_midpoint(0.5j, -0.5j), 0)
        self.assertAlmostEqual(geodesic_midpoint(0.6, 0.6j), 0.24764464962+0.247644649627j)

    # to be extended