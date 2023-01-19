import unittest
from hypertiling import HyperbolicTiling

print("Testing different neighbour search algorithms against each other")


class TestCore(unittest.TestCase):
    def test_comp_nbrs(self):

        kernel = "SR"

        lattices = [(3, 7, 6,), (7, 3, 6), (5, 4, 5), (4, 5, 5), (9, 3, 3), (4, 10, 3), (3, 8, 3), (6, 4, 3)]

        for p, q, nlayer in lattices:
            print("Constructing", p, q, nlayer, "lattice")
            T = HyperbolicTiling(p, q, nlayer, kernel=kernel, center="cell")

            nbrs1 = T.get_nbrs_list(method="RBF").sort()
            nbrs2 = T.get_nbrs_list(method="RO").sort()
            nbrs3 = T.get_nbrs_list(method="EMO").sort()

            self.assertEqual(nbrs1, nbrs2, nbrs3)


        kernel = "SRL"

        lattices = [(3, 7, 4), (7, 3, 4), (5, 4, 4), (4, 5, 4), (9, 3, 4), (4, 10, 3), (3, 8, 4), (6, 4, 4)]

        for p, q, nlayer in lattices:
            print("Constructing", p, q, nlayer, "lattice")
            T = HyperbolicTiling(p, q, nlayer, kernel=kernel, center="cell")

            nbrs1 = T.get_nbrs_list(method="RBF").sort()
            nbrs2 = T.get_nbrs_list(method="RO").sort()
            nbrs3 = T.get_nbrs_list(method="EMO").sort()

            self.assertEqual(nbrs1, nbrs2, nbrs3)


if __name__ == '__main__':
    unittest.main()

