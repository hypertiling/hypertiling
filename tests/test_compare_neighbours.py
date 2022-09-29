import unittest
from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find

print("Testing different neighbour search algorithms against each other")


class TestCore(unittest.TestCase):
    def test_comp_nbrs(self):

        kernel = "SRI"

        lattices = [(3, 7, 7,), (7, 3, 7), (5, 4, 6), (4, 5, 6), (9, 3, 4), (4, 10, 3), (3, 8, 4), (6, 4, 4)]

        for p, q, nlayer in lattices:
            print("Constructing", p, q, nlayer, "lattice")
            T = HyperbolicTiling(p, q, nlayer, kernel=kernel, center="cell")
            T.generate()

            nbrs1 = find(T, which="radius-optimized")
            nbrs2 = T.get_nbrs_radius_optimized_slice()

            self.assertEqual(nbrs1, nbrs2)

        kernel = "SRI"

        lattices = [(3, 7, 4), (7, 3, 4), (5, 4, 4), (4, 5, 4), (9, 3, 4), (4, 10, 3), (3, 8, 4), (6, 4, 4)]

        for p, q, nlayer in lattices:
            print("Constructing", p, q, nlayer, "lattice")
            T = HyperbolicTiling(p, q, nlayer, kernel=kernel, center="cell")
            T.generate()

            nbrs1 = find(T, which="radius-optimized")
            nbrs2 = find(T, which="brute-force-radius")

            self.assertEqual(nbrs1, nbrs2)


if __name__ == '__main__':
    unittest.main()
