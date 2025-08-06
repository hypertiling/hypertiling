import unittest
from hypertiling.core import HyperbolicTiling, TilingKernels, HyperbolicGraph, GraphKernels

pqrns = [
    (5, 4, 2, 25),
    (5, 5, 2, 23),
    (7, 3, 2, 26),
    (2, 3, 7, 20),
    (4, 4, 4, 15),
    (11, 4, 13, 11),
]

lengths = [11008, 28232, 3048, 1610, 27728, 20098]


class TestOperators(unittest.TestCase):

    def test_init(self):

        # test for sectors
        for (p, q, r, n), length in zip(pqrns, lengths):
            t = HyperbolicTiling(p, q, r, n, kernel=TilingKernels.GRCT, nbrs=True, size=50_000)
            g = HyperbolicGraph(p, q, r, n, kernel=GraphKernels.GRCT, tiling=False, size=50_000)

            # test basic properties
            self.assertEqual(length, len(t))

            # special methods
            self.assertEqual(t[0][3], 0.0)
            self.assertEqual(len(g[0]), 3)
            for verts in t:
                self.assertEqual(verts[3], 0.0)
                break

            # normal methods
            self.assertEqual(4, len(t.get_vertices(0)))
            self.assertEqual(4, len(t.get_vertices(len(t) - 1)))

            self.assertEqual(3, len(t.get_nbrs(0)))

            self.assertEqual(0, t.get_reflection_level(0))
            self.assertEqual(n - 1, t.get_reflection_level(len(t.coords) - 1))

            nbrs_list = t.get_nbrs_list()
            self.assertEqual(len(t), len(nbrs_list))
            self.assertEqual(3, len(nbrs_list[0]))

            # check integrity of tesselation
            t.check_integrity()

    def test_raise(self):
        # test for missing neighbors
        t = HyperbolicTiling(5, 4, 2, 10, kernel=TilingKernels.GRCT, size=50_000)

        with self.assertRaises(AttributeError):
            t.get_nbrs(0)

        with self.assertRaises(AttributeError):
            t.get_nbrs_list()

        with self.assertRaises(AttributeError):
            t.check_integrity()

        # test for missing coordinates
        g = HyperbolicGraph(5, 4, 2, 10, kernel=GraphKernels.GRCT, tiling=False, size=50_000)

        with self.assertRaises(AttributeError):
            for i in g:
                pass

        with self.assertRaises(AttributeError):
            g.get_vertices(0)


if __name__ == '__main__':
    unittest.main()