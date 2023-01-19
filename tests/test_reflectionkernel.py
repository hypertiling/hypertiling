import unittest
from tests.test_util import *
from hypertiling.core import HyperbolicTiling
import hypertiling.geodesics as geos
import hypertiling.kernels.GR_util as util
import random
import numpy as np

# overall
COMBIS = [(7, 3, 4),
          (3, 7, 4),
          (5, 4, 3),
          (4, 5, 3),
          (6, 4, 3),
          (7, 4, 3),
          (7, 5, 3),
          (3, 8, 3)]

# test generate
MAXLAYERS = 4

# test check_integrity
RATIOOFDUPLICATES = 0.1
RATIOOFHOLES = 0.1

# test find/get_nbrs_generative
SHIFTTOL = 1e-5


class TestReflectTiling(unittest.TestCase):

    def test_generate(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(combi[0], combi[1], n=MAXLAYERS, kernel="GR")
            # this is basically the only test we can do and it only will scream when duplicates are found
            with PrintTest() as stream:
                tiling.check_integrity()

    def test_find(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")

            for index in range(tiling.length - 1):
                center = tiling[index][0]

                # find with center
                self.assertEqual(tiling.find(center), index)

                # test boundary of findings
                neighbor_centers = util.generate_raw(tiling[index])
                for neighbor in neighbor_centers:
                    midpoint = geos.geodesic_midpoint(neighbor, center)
                    v = (center - midpoint) * SHIFTTOL
                    test_point = midpoint + v
                    self.assertEqual(tiling.find(test_point), index)

                # test non finding
                if 1 < index < len(tiling._sector_polys):  # 0 and 1 are excluded as they are used as reference in find
                    tiling._sector_polys[index, 0] = 0 if index != 0 else 0.999999999999j
                    self.assertFalse(tiling.find(center))
                    tiling._sector_polys[index, 0] = center

    def test_get_neighbors(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")
            for index in range(tiling.length):
                neighbors = tiling.get_nbrs_generative(index)

                for j in range(tiling.length):
                    if j == index:
                        continue
                    poly = tiling[j]

                    midpoint = geos.geodesic_midpoint(tiling[index][0], poly[0])
                    v = (tiling[index][0] - midpoint) * SHIFTTOL
                    testpoint = midpoint + v
                    found = tiling.find(testpoint)

                    if j in neighbors:
                        self.assertEqual(index, found)

                    elif found is not False:
                        """
                        If the polygons are not neighbors, the algorithm will either 
                        1. find no polygon when searching close to the middle of both (return False)
                        2. Will return another polygon which should not be index
                        """
                        self.assertNotEqual(index, found)

    def test_get_neighbors_radius(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")
                tiling.map_nbrs()
            for index in range(tiling.length):
                neighbors = tiling.get_nbrs_mapping(index)
                neighbors2 = tiling.get_nbrs_radius(index)
                try:
                    self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))
                except Exception as error:
                    print(index)
                    print(combi)
                    print(neighbors)
                    print(neighbors2)
                    raise error

    def test_get_neighbors_experimental(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")
                tiling.map_nbrs()
            for index in range(tiling.length):
                neighbors = tiling.get_nbrs_mapping(index)
                neighbors2 = tiling.get_nbrs_geometrical(index)
                self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))

    def test_get_neighbors_mapping(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")
            tiling.map_nbrs()
            for index in range(tiling.length):
                neighbors = tiling.get_nbrs_generative(index)
                neighbors2 = tiling.get_nbrs_mapping(index)
                self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))

    def test_get_neighbors_list(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")
            neighbors_list = tiling.get_nbrs_list()  # calls tiling.map_nbrs
            for index in range(tiling.length):
                neighbors = neighbors_list[index]
                neighbors2 = tiling.get_nbrs_mapping(index)
                self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))

    def test_check_integrity(self):
        for combi in Progress(COMBIS):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel="GR")

            # check for duplicates
            old = tiling._sector_polys[-1, 0]
            for i in range(int(RATIOOFDUPLICATES * len(tiling._sector_polys))):
                index = random.randint(2, len(tiling._sector_polys) - 2)  # ignore last as it will be used as duplicate
                tiling._sector_polys[-1, 0] = tiling._sector_polys[index, 0]
                with self.assertRaises(AttributeError, msg=f"Duplicate at index {index} was not detected") as error:
                    tiling.check_integrity()
                self.assertEqual(f"Duplicate detected at index {index} at layer {tiling.get_reflection_level(index)}",
                                 str(error.exception))
            tiling._sector_polys[-1, 0] = old

        print("Testing for holes is not written yet (updated)")
        """ # check layer boundary/holes
        indices = np.argwhere(tiling._layers < boundary_layer)[1:]  # eliminate first polygon
        for i in range(int(RATIOOFHOLES * len(tiling._sector_polys))):
            index = random.choice(indices)[0]
            old = tiling._sector_polys[index, 0]
            tiling._sector_polys[index, 0] = 0
            with PrintTest() as stream:
                tiling.check_integrity()
            tiling._sector_polys[index, 0] = old
            values = stream.get()
            messages = values[:-1].split("\n")
            last_layer = int(messages[-1].split(" ")[-1])
            self.assertIn(last_layer, [tiling._layers[index] - 1, tiling._layers[index]])"""


if __name__ == '__main__':
    tiling = HyperbolicTiling(7, 3, 2, kernel="GR")
    unittest.main()
