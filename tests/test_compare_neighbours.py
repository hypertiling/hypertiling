import unittest
from hypertiling import HyperbolicTiling
from hypertiling.neighbours import find


kernel = "flo"

lattices = [(3,7,7,), (7,3,7), (5,4,6), (4,5,6), (9,3,4), (4,10,3), (3,8,4), (6,4,4)]

print("Testing different neighbour search algorithms against each other")

class TestCompareNeighbours(unittest.TestCase):
    def test_comp_nbrs(self):

        for p, q, nlayer in lattices:
            print("Constructing", p, q, nlayer, "lattice")
            T = HyperbolicTiling(p, q, nlayer, kernel=kernel, center="cell")
            T.generate()

            nbrs1 = find(T, which="optimized")
            nbrs2 = find(T, which="optimized_slice")
            
            self.assertTrue(nbrs1 == nbrs2)


if __name__ == '__main__':
    unittest.main()
