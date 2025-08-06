import unittest
from hypertiling.core import HyperbolicTiling, TilingKernels, HyperbolicGraph, GraphKernels

pqs = [
    (3, 7, 15),
    (3, 8, 16),
    (3, 12, 15),
    (4, 5, 10),
    (5, 4, 9),
    (6, 4, 8),
    (7, 3, 9),
    (7, 4, 7),
    (7, 5, 6),
    (14, 3, 5)
]

lengths = [4417, 19939, 39325, 7285, 7981, 23857, 11173, 23353, 9311, 15261]


class TestOperators(unittest.TestCase):

    def test_initSectorTrue(self):
        # test for sectors
        for (p, q, n), length in zip(pqs, lengths):
            t = HyperbolicTiling(p, q, n, kernel=TilingKernels.GRC, nbrs=True, sector=True)
            g = HyperbolicGraph(p, q, n, kernel=GraphKernels.GRC, tiling=False, sector=True)

            # test basic properties
            self.assertEqual(length, len(t))

            # special methods
            self.assertEqual(t[0][0], 0.0)
            self.assertEqual(len(g[0]), p)
            for verts in t:
                self.assertEqual(verts[0], 0.0)
                break

            # normal methods
            p1 = p + 1
            self.assertEqual(p1, len(t.get_vertices(0)))
            self.assertEqual(p1, len(t.get_vertices(len(t) - 1)))

            self.assertEqual(p, len(t.get_nbrs(0)))
            self.assertEqual(p, len(t.get_nbrs(len(t.coords) + 1)))

            self.assertEqual(0, t.get_reflection_level(0))
            self.assertEqual(n - 1, t.get_reflection_level(len(t.coords) - 1))
            self.assertEqual(1, t.get_reflection_level(len(t.coords)))

            nbrs_list = t.get_nbrs_list()
            self.assertEqual(len(t), len(nbrs_list))
            self.assertEqual(p, len(nbrs_list[0]))
            self.assertEqual(p, len(nbrs_list[len(t.coords)]))

            # check integrity of tesselation
            t.check_integrity()

    def test_initSectorFalse(self):

        # test for sectors
        for (p, q, n), length in zip(pqs, lengths):
            t = HyperbolicTiling(p, q, n, kernel=TilingKernels.GRC, nbrs=True, sector=False)

            # test basic properties
            self.assertEqual(length, len(t))
            self.assertEqual(length, len(t.coords))

            # special methods
            self.assertEqual(t[0][0], 0.0)
            for verts in t:
                self.assertEqual(verts[0], 0.0)
                break

            # normal methods
            p1 = p + 1
            self.assertEqual(p1, len(t.get_vertices(0)))
            self.assertEqual(p1, len(t.get_vertices(len(t) - 1)))

            self.assertEqual(p, len(t.get_nbrs(0)))

            self.assertEqual(0, t.get_reflection_level(0))
            self.assertEqual(n - 1, t.get_reflection_level(len(t.coords) - 1))

            nbrs_list = t.get_nbrs_list()
            self.assertEqual(len(t), len(nbrs_list))
            self.assertEqual(p, len(nbrs_list[0]))

            # check integrity of tesselation
            t.check_integrity()

    def test_raise(self):
        # test for missing neighbors
        t = HyperbolicTiling(3, 7, 5, kernel=TilingKernels.GRC, sector=True, nbrs=False)
        with self.assertRaises(AttributeError):
            t.get_nbrs(0)

        with self.assertRaises(AttributeError):
            t.get_nbrs_list()

        with self.assertRaises(AttributeError):
            t.check_integrity()

        # test for missing coordinates
        t = HyperbolicGraph(3, 7, 5, kernel=GraphKernels.GRC, sector=True)
        with self.assertRaises(AttributeError):
            for i in t:
                pass

        with self.assertRaises(AttributeError):
            t.get_vertices(0)


if __name__ == '__main__':
    unittest.main()