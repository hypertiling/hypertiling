import unittest
from tests.test_util import *
from hypertiling.core import HyperbolicTiling, TilingKernels
from hypertiling.kernel.GR import GenerativeReflection
from hypertiling.kernel_abc import Tiling
import hypertiling.geodesics as geos
import hypertiling.kernel.GR_util as util
import random
import numpy as np

# overall
MAXLAYER = 4
min_layer = lambda p, q: q + 1 if q & 1 else q // 2 + 1  # this is necessary for 1st and 2nd order Fillers to appear


combis = [(7, 3),
          (3, 7),
          (5, 4),
          (4, 5),
          (6, 4),
          (7, 4),
          (7, 5),
          (3, 8)]

combis = [(combi[0], combi[1], min(min_layer(*combi), MAXLAYER)) for combi in combis]
# test check_integrity !stochastic test!
RATIOOFDUPLICATES = 0.1
RATIOOFHOLES = 0.1

# test find/get_nbrs_generative
SHIFTTOL = 1e-5


class TestReflectTiling(unittest.TestCase):

    def test_generate(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(combi[0], combi[1], n=combi[2], kernel=TilingKernels.GenerativeReflection)

            # check if the tiling is from the correct kernel
            self.assertTrue(isinstance(tiling, Tiling))
            self.assertTrue(isinstance(tiling, GenerativeReflection))
            # this is basically the only test we can do and it only will scream when duplicates are found
            with PrintTest() as stream:
                tiling.check_integrity()

    def test_find(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)

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
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)
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
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)
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

    def test_get_neighbors_geometrical(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)
                tiling.map_nbrs()
            for index in range(tiling.length):
                neighbors = tiling.get_nbrs_mapping(index)
                neighbors2 = tiling.get_nbrs_geometrical(index)
                self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))

    def test_get_neighbors_mapping(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)
            tiling.map_nbrs()
            for index in range(tiling.length):
                neighbors = tiling.get_nbrs_generative(index)
                neighbors2 = tiling.get_nbrs_mapping(index)
                self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))

    def test_get_neighbors_list(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)
            neighbors_list = tiling.get_nbrs_list()  # calls tiling.map_nbrs
            for index in range(tiling.length):
                neighbors = neighbors_list[index]
                neighbors2 = tiling.get_nbrs_mapping(index)
                self.assertTrue(np.array_equal(np.sort(neighbors2), np.sort(neighbors)))

    def test_check_integrity(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)

            # check for duplicates
            old = tiling._sector_polys[-1, 0]
            for i in range(int(RATIOOFDUPLICATES * len(tiling._sector_polys))):
                index = random.randint(2, len(tiling._sector_polys) - 2)  # ignore last as it will be used as duplicate
                tiling._sector_polys[-1, 0] = tiling._sector_polys[index, 0]
                with self.assertRaises(AttributeError, msg=f"Duplicate at index {index} was not detected") as error:
                    tiling.check_integrity()
                self.assertEqual(f"[hypertiling] Error: Duplicate detected at index {index} at layer {tiling.get_reflection_level(index)}",
                                 str(error.exception))
            tiling._sector_polys[-1, 0] = old

        # check layer boundary/holes
        for i in range(int(RATIOOFHOLES * len(tiling))):
            index = random.randint(1, tiling._sector_polys.shape[0] - 1)  # first one is excluded as it would be at 0
            old = tiling._sector_polys[index, 0]
            tiling._sector_polys[index, 0] = 0  # move center to 0
            with PrintTest() as stream:
                tiling.check_integrity()
            tiling._sector_polys[index, 0] = old
            values = stream.get()
            messages = values[:-1].split("\n")
            last_layer = int(messages[-1].split(" ")[-1])
            reflection_layer = tiling.get_reflection_level(index)
            self.assertIn(last_layer, [reflection_layer - 1, reflection_layer])

    def test_get_reflection_levels(self):
        for combi in Progress(combis):
            with PrintTest():
                tiling = HyperbolicTiling(*combi, kernel=TilingKernels.GenerativeReflection)

            last_layer = 0
            for index in range(tiling._sector_polys.shape[0]):
                rl = tiling.get_reflection_level(index)
                self.assertGreaterEqual(rl, last_layer)
                last_layer = rl
            self.assertEqual(last_layer + 1, combi[2])


if __name__ == '__main__':
    unittest.main()
