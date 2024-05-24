import unittest
from hypertiling import HyperbolicTiling
from hypertiling import TilingKernels
from hypertiling.util import n_cell_centered, n_vertex_centered

class TestCore(unittest.TestCase):
    def test_num_cells_cell_centered(self):
        maxlayer = 4
        
        kernels = [TilingKernels.StaticRotationalSector, TilingKernels.StaticRotationalGraph]
        
        for k in kernels:

            for p,q in [(7,3), (8,3), (4,5), (4,6), (4,7), (4,8), (5,5), (5,6)]:

                for n in range(1, maxlayer):
                    print("Constructing lattice (p,q,n) = ", p, q, n)
                    T = HyperbolicTiling(p, q, n, kernel=k)
                    self.assertEqual(n_cell_centered(p, q, n), len(T))

                    print("Constructing lattice (p,q,n) = ", q, p, n)
                    T = HyperbolicTiling(q, p, n, kernel=k)
                    self.assertEqual(n_cell_centered(q, p, n), len(T))

        for k in kernels:

            for p,q in [(7,3), (8,3), (4,5), (4,6), (4,7), (4,8), (5,5), (5,6)]:

                for n in range(1, maxlayer):
                    print("Constructing lattice (p,q,n) = ", p, q, n)
                    T = HyperbolicTiling(p, q, n, center="vertex", kernel=k)
                    self.assertEqual(n_vertex_centered(p, q, n), len(T))

                    print("Constructing lattice (p,q,n) = ", q, p, n)
                    T = HyperbolicTiling(q, p, n, center="vertex", kernel=k)
                    self.assertEqual(n_vertex_centered(q, p, n), len(T))


if __name__ == '__main__':
    unittest.main()
