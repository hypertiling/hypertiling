import numpy as np
import unittest

from hypertiling.arithmetics import kahan, twosum, twodiff, twoproduct

class TestKahan(unittest.TestCase):
    
    def test_positive_float_addition(self):
        result = kahan(3.5, 2.5)
        self.assertAlmostEqual(result[0], 6.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_array_addition(self):
        x = np.array([1.2, 2.3, 3.4])
        y = np.array([0.5, 1.5, 2.5])
        result = kahan(x, y)
        np.testing.assert_array_almost_equal(result[0], np.array([1.7, 3.8, 5.9]))
        np.testing.assert_array_almost_equal(result[1], np.array([0.0, 0.0, 0.0]))

    def test_positive_float_addition_float32(self):
        result = kahan(np.float32(3.5), np.float32(2.5))
        self.assertAlmostEqual(result[0], 6.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_array_addition_float32(self):
        x = np.array([1.2, 2.3, 3.4], dtype=np.float32)
        y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        result = kahan(x, y)
        np.testing.assert_array_almost_equal(result[0], np.array([1.7, 3.8, 5.9]), decimal=6)
        np.testing.assert_array_almost_equal(result[1], np.array([0.0, 0.0, 0.0]), decimal=6)




class TestTwosum(unittest.TestCase):
    def test_float64_numbers(self):
        result, error = twosum(3.5, 2.5)
        self.assertAlmostEqual(result, 6.0)
        self.assertAlmostEqual(error, 0.0)

    def test_float64_arrays(self):
        x = np.array([1.1, 2.2, 3.3])
        y = np.array([0.5, 1.5, 2.5])
        result, error = twosum(x, y)
        np.testing.assert_array_almost_equal(result, np.array([1.6, 3.7, 5.8]))
        np.testing.assert_array_almost_equal(error, np.array([0.0, 0.0, 0.0]))

    def test_float32_numbers(self):
        result, error = twosum(np.float32(3.5), np.float32(2.5))
        self.assertAlmostEqual(result, 6.0)
        self.assertAlmostEqual(error, 0.0)

    def test_float32_arrays(self):
        x = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        result, error = twosum(x, y)
        np.testing.assert_array_almost_equal(result, np.array([1.6, 3.7, 5.8], dtype=np.float32))
        np.testing.assert_array_almost_equal(error, np.array([0.0, 0.0, 0.0], dtype=np.float32))


class TestTwodiff(unittest.TestCase):
    def test_float64(self):
        x = 5.0
        y = 3.0
        expected_result = (2.0, 0.0)
        self.assertEqual(twodiff(x, y), expected_result)

    def test_float32(self):
        x = np.float32(5.0)
        y = np.float32(3.0)
        expected_result = (np.float32(2.0), np.float32(0.0))
        self.assertTrue(np.allclose(twodiff(x, y), expected_result))

    def test_float64_arrays(self):
        x = np.array([5.0, 6.0])
        y = np.array([3.0, 2.0])
        expected_result = (np.array([2.0, 4.0]), np.array([0.0, 0.0]))
        np.testing.assert_array_equal(twodiff(x, y), expected_result)

    def test_float32_arrays(self):
        x = np.array([5.0, 6.0], dtype=np.float32)
        y = np.array([3.0, 2.0], dtype=np.float32)
        expected_result = (np.array([2.0, 4.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(twodiff(x, y), expected_result)

class TestTwoProduct(unittest.TestCase):
    def test_basic_multiplication(self):
        x = 5.0
        y = 2.0
        result, error = twoproduct(x, y)
        self.assertAlmostEqual(result, 10.0)
        self.assertAlmostEqual(error, 0.0)

    def test_array_multiplication(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        result, error = twoproduct(x, y)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 6.0, 12.0]))
        np.testing.assert_array_almost_equal(error, np.array([0.0, 0.0, 0.0]))

    def test_number_and_array_multiplication(self):
        x = 2.0
        y = np.array([2.0, 3.0, 4.0])
        result, error = twoproduct(x, y)
        np.testing.assert_array_almost_equal(result, np.array([4.0, 6.0, 8.0]))
        np.testing.assert_array_almost_equal(error, np.array([0.0, 0.0, 0.0]))

    def test_array_and_number_multiplication(self):
        x = np.array([2.0, 3.0, 4.0])
        y = 2.0
        result, error = twoproduct(x, y)
        np.testing.assert_array_almost_equal(result, np.array([4.0, 6.0, 8.0]))
        np.testing.assert_array_almost_equal(error, np.array([0.0, 0.0, 0.0]))




if __name__ == '__main__':
    unittest.main()