import unittest
import numpy as np
from hypertiling import HyperbolicTiling

# These tests closely follow the dynamic-manipulation demo notebook

def my_angular_filter(z):
    return True if (45 < np.angle(z, deg=True) < 180) else False


class TestStaticRotationalGraph(unittest.TestCase):

    def test_addition_and_removal(self):

        T = HyperbolicTiling(7, 3, 2, kernel="SRG", center="vertex")
        self.assertEqual(len(T), 15)
        # ----

        T.add_layer()
        self.assertEqual(len(T), 48)
        # ----

        T.remove_cells([7,8,9,10,11,12,13,14])
        T.remove_cells([1])
        T.remove_cells(range(27,36))
        T.remove_cells(range(37,42))
        self.assertEqual(len(T), 25)
        # ----

        T.add_layer([0,36])
        self.assertEqual(len(T), 34)


    def test_filtering(self):

        T = HyperbolicTiling(5,4, 2, kernel="SRG")
        T.add_layer(filter = my_angular_filter)
        T.add_layer(filter = my_angular_filter)
        T.add_layer(filter = my_angular_filter)

        self.assertEqual(len(T), 288)


if __name__ == '__main__':
    unittest.main()
