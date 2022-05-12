import unittest
from hypertiling import HyperbolicTiling
from hypertiling.util import n_cell_centered



class TestCore(unittest.TestCase):
    def test_num_cells_cell_centered(self):
        l = 7
        p, q = 3, 7
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 3, 8
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 4, 5
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 4, 6
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 4, 7
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 4, 8
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 5, 5
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))

        p, q = 5, 6
        for n in range(l):
            print("Constructing lattice (p,q,n) = ", p, q, n)
            T = HyperbolicTiling(p, q, n)
            T.generate()
            self.assertEqual(n_cell_centered(p,q,n),len(T))

            print("Constructing lattice (p,q,n) = ", q, p, n)
            T = HyperbolicTiling(q, p, n)
            T.generate()
            self.assertEqual(n_cell_centered(q,p,n),len(T))



if __name__ == '__main__':
    unittest.main()
