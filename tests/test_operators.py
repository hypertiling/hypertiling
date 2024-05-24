<<<<<<< HEAD
import unittest
from hypertiling.operators import *

class TestOperators(unittest.TestCase):

    def test_adjacency(self):
        neighbours = [[1,2],[0,2],[0,1]]
        adj = adjacency(neighbours)
        self.assertEqual(adj.shape,(3,3))
        self.assertEqual(adj.nnz,6)
        self.assertEqual(adj.todense()[0,0],0)
        self.assertEqual(adj.todense()[0,1],1)
        self.assertEqual(adj.todense()[0,2],1)
        self.assertEqual(adj.todense()[1,0],1)
        self.assertEqual(adj.todense()[1,1],0)
        self.assertEqual(adj.todense()[1,2],1)
        self.assertEqual(adj.todense()[2,0],1)
        self.assertEqual(adj.todense()[2,1],1)
        self.assertEqual(adj.todense()[2,2],0)

    def test_adjacency_weights(self):
        neighbours = [[1,2],[0,2],[0,1]]
        weights = [[1,2],[3,4],[5,6]]
        adj = adjacency(neighbours,weights=weights)
        self.assertEqual(adj.shape,(3,3))
        self.assertEqual(adj.nnz,6)
        self.assertEqual(adj.todense()[0,0],0)
        self.assertEqual(adj.todense()[0,1],1)
        self.assertEqual(adj.todense()[0,2],2)
        self.assertEqual(adj.todense()[1,0],3)
        self.assertEqual(adj.todense()[1,1],0)
        self.assertEqual(adj.todense()[1,2],4)
        self.assertEqual(adj.todense()[2,0],5)
        self.assertEqual(adj.todense()[2,1],6)
        self.assertEqual(adj.todense()[2,2],0)


    def test_adjacency_boundary(self):
        neighbours = [[1,2],[0,2],[0,1]]
        boundary = [False,False,True]
        adj = adjacency(neighbours,boundary=boundary)
        self.assertEqual(adj.shape,(3,3))
        self.assertEqual(adj.nnz,4)
        self.assertEqual(adj.todense()[0,0],0)
        self.assertEqual(adj.todense()[0,1],1)
        self.assertEqual(adj.todense()[0,2],1)
        self.assertEqual(adj.todense()[1,0],1)
        self.assertEqual(adj.todense()[1,1],0)
        self.assertEqual(adj.todense()[1,2],1)
        self.assertEqual(adj.todense()[2,0],0)
        self.assertEqual(adj.todense()[2,1],0)
        self.assertEqual(adj.todense()[2,2],0)


    def test_degree(self):
        neighbours = [[1,2],[0,2],[0,1]]
        deg = degree(neighbours)
        self.assertEqual(deg.shape,(3,3))
        self.assertEqual(deg.nnz,3)
        self.assertEqual(deg.todense()[0,0],2)
        self.assertEqual(deg.todense()[0,1],0)
        self.assertEqual(deg.todense()[0,2],0)
        self.assertEqual(deg.todense()[1,0],0)
        self.assertEqual(deg.todense()[1,1],2)
        self.assertEqual(deg.todense()[1,2],0)
        self.assertEqual(deg.todense()[2,0],0)
        self.assertEqual(deg.todense()[2,1],0)
        self.assertEqual(deg.todense()[2,2],2)

    def test_degree_weights(self):
        neighbours = [[1,2],[0,2],[0,1]]
        weights = [[1,2],[3,4],[5,6]]
        deg = degree(neighbours,weights=weights)
        self.assertEqual(deg.shape,(3,3))
        self.assertEqual(deg.nnz,3)
        self.assertEqual(deg.todense()[0,0],3)
        self.assertEqual(deg.todense()[0,1],0)
        self.assertEqual(deg.todense()[0,2],0)
        self.assertEqual(deg.todense()[1,0],0)
        self.assertEqual(deg.todense()[1,1],7)
        self.assertEqual(deg.todense()[1,2],0)
        self.assertEqual(deg.todense()[2,0],0)
        self.assertEqual(deg.todense()[2,1],0)
        self.assertEqual(deg.todense()[2,2],11)

    def test_degree_boundary(self):
        neighbours = [[1,2],[0,2],[0,1]]
        boundary = [False,False,True]
        deg = degree(neighbours,boundary=boundary)
        self.assertEqual(deg.shape,(3,3))
        self.assertEqual(deg.nnz,2)
        self.assertEqual(deg.todense()[0,0],2)
        self.assertEqual(deg.todense()[0,1],0)
        self.assertEqual(deg.todense()[0,2],0)
        self.assertEqual(deg.todense()[1,0],0)
        self.assertEqual(deg.todense()[1,1],2)
        self.assertEqual(deg.todense()[1,2],0)
        self.assertEqual(deg.todense()[2,0],0)
        self.assertEqual(deg.todense()[2,1],0)
        self.assertEqual(deg.todense()[2,2],0)

    def test_identity(self):
        neighbours = [[1,2],[0,2],[0,1]]
        id = identity(neighbours)
        self.assertEqual(id.shape,(3,3))
        self.assertEqual(id.nnz,3)
        self.assertEqual(id.todense()[0,0],1)
        self.assertEqual(id.todense()[0,1],0)
        self.assertEqual(id.todense()[0,2],0)
        self.assertEqual(id.todense()[1,0],0)
        self.assertEqual(id.todense()[1,1],1)
        self.assertEqual(id.todense()[1,2],0)
        self.assertEqual(id.todense()[2,0],0)
        self.assertEqual(id.todense()[2,1],0)
        self.assertEqual(id.todense()[2,2],1)


    def test_helmholtz(self):
        neighbours = [[1,2],[0,2],[0,1]]
        helm = helmholtz_from_hypergraph_sparse(neighbours, 5, 7, 2)
        self.assertEqual(helm.shape,(3,3))
        self.assertEqual(helm.nnz,15)
        self.assertEqual(helm.todense()[0,0],-19)
        self.assertEqual(helm.todense()[0,1],2)
        self.assertEqual(helm.todense()[0,2],2)
=======
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
>>>>>>> dev
