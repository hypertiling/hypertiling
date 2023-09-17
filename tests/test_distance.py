import unittest
from hypertiling.distance import lorentzian_distance, weierstrass_distance, disk_distance
from hypertiling.representations import valid_weierstrass_point

class TestDistance(unittest.TestCase):
    """
    This class 
    """

    def test_valid_weierstrass_point(self):
        self.assertFalse(valid_weierstrass_point([1, 0, 0]))
        self.assertFalse(valid_weierstrass_point([0, 0, 0]))
        self.assertTrue(valid_weierstrass_point([2, 1.5, 1]))
        self.assertTrue(valid_weierstrass_point([3, 2, 1.5]))
        self.assertTrue(valid_weierstrass_point([2.5, 1.0, 0.5]))


    def test_distance(self):
        sample_point1 = [2, 1.5, 1]
        sample_point2 = [3, 2, 1.5]
        sample_point3 = [2.5, 1.0, 0.5]


        # Test that the distance between a point and itself is zero
        self.assertEqual(lorentzian_distance(sample_point1,sample_point1), 0.75) # Lorentzian distance is actually more a norm
        self.assertEqual(weierstrass_distance(sample_point1,sample_point1), 0)
        self.assertEqual(disk_distance(0.5, 0.5), 0)

        # Test that the distance between two points is positive
        self.assertGreater(lorentzian_distance(sample_point1, sample_point2), 0)
        self.assertGreater(weierstrass_distance(sample_point1, sample_point2), 0)
        self.assertGreater(disk_distance(0.3, 0.5), 0)

        # Test that the distance between two points is symmetric
        self.assertEqual(lorentzian_distance(sample_point1, sample_point2), lorentzian_distance(sample_point2, sample_point1))
        self.assertEqual(weierstrass_distance(sample_point1, sample_point2), weierstrass_distance(sample_point2, sample_point1))
        self.assertEqual(disk_distance(0.3, 0.5), disk_distance(0.5, 0.3))

        # Test triangle inequality
        self.assertLessEqual(lorentzian_distance(sample_point1, sample_point2), lorentzian_distance(sample_point1, sample_point3)+lorentzian_distance(sample_point3, sample_point2))
        self.assertLessEqual(weierstrass_distance(sample_point1, sample_point2), weierstrass_distance(sample_point1, sample_point3)+weierstrass_distance(sample_point3, sample_point2))
        self.assertLessEqual(disk_distance(0.3, 0.5), disk_distance(0.3, 0.7)+disk_distance(0.7, 0.5))
