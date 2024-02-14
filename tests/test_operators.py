import numpy as np
import scipy.sparse as sps
import unittest
from hypertiling.operators import adjacency, degree, identity, helmholtz_from_hypergraph_sparse

class TestAdjacency(unittest.TestCase):
    def test_no_weights_no_boundary(self):
        neighbours = [[1, 2], [0, 2], [0, 1]]
        result = adjacency(neighbours)
        expected_result = sps.coo_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        self.assertTrue(np.array_equal(result.toarray(), expected_result.toarray()))

    def test_with_weights_no_boundary(self):
        neighbours = [[1, 2], [0, 2], [0, 1]]
        weights = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        result = adjacency(neighbours, weights)
        expected_result = sps.coo_matrix([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        self.assertTrue(np.array_equal(result.toarray(), expected_result.toarray()))



# def test_degree():
#     neighbours = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
#     weights = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0]]
#     boundary = [True, False, False, True]

#     # Testing for the case when weights and boundary are both None
#     result1 = degree(neighbours)
#     assert np.array_equal(result1.toarray(), np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]]), "Test case 1 failed")

#     # Testing for the case when weights is provided but boundary is None
#     result2 = degree(neighbours, weights)
#     assert np.array_equal(result2.toarray(), np.array([[3, 0, 0, 0], [0, 12, 0, 0], [0, 0, 21, 0], [0, 0, 0, 19]]), "Test case 2 failed")

#     # Testing for the case when both weights and boundary are provided
#     result3 = degree(neighbours, weights, boundary)
#     assert np.array_equal(result3.toarray(), np.array([[0, 0, 0, 0], [0, 12, 0, 0], [0, 0, 21, 0], [0, 0, 0, 0]]), "Test case 3 failed")

#     # Testing for the case when neighbours is an empty list
#     result4 = degree([], None, None)
#     assert np.array_equal(result4.toarray(), np.array([]), "Test case 4 failed")




# def test_identity():
#     # Test with only neighbours argument provided
#     neighbours = [[1, 2], [0, 2], [0, 1]]
#     result = identity(neighbours)
#     assert isinstance(result, sps.coo_matrix)
#     assert result.shape == (3, 3)
#     assert np.array_equal(result.toarray(), np.identity(3))

#     # Test with neighbours and weights arguments provided
#     weights = [[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]]
#     result = identity(neighbours, weights)
#     assert isinstance(result, sps.coo_matrix)
#     assert result.shape == (3, 3)
#     expected_array = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
#     assert np.allclose(result.toarray(), expected_array)

#     # Test with neighbours, weights, and boundary arguments provided
#     boundary = [False, False, True]
#     result = identity(neighbours, weights, boundary)
#     assert isinstance(result, sps.coo_matrix)
#     assert result.shape == (3, 3)
#     expected_array = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
#     assert np.allclose(result.toarray(), expected_array)




class TestHelmholtzFromHypergraphSparse(unittest.TestCase):
    
    def test_empty_neighbours(self):
        result = helmholtz_from_hypergraph_sparse([], 1, 1, 1)
        self.assertEqual(result.nnz, 0)

    # add further cases for this function ...


if __name__ == '__main__':
    unittest.main()