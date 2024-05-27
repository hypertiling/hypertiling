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
