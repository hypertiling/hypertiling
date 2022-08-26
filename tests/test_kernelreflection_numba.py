import unittest
from tests.test_util import *
from hypertiling.array.kernelreflection_numba import ReflectTiling
import hypertiling.geodesics as geos
import hypertiling.array.reflection_numba_util as util
import random
import numpy as np

# overall
COMBIS = [(7, 3, 6),
          (3, 7, 4),
          (5, 4, 6),
          (4, 5, 6),
          (6, 4, 4),
          (7, 4, 4),
          (7, 5, 3),
          (7, 6, 3),
          (7, 7, 3),
          (3, 8, 4),
          (8, 3, 4)]

# test generate
MAXLAYERS = 10

# test check_integrity
RATIOOFDUPLICATES = 0.1
RATIOOFHOLES = 0.1

# test find/get_neighbors
SHIFTTOL = 1e-5


class TestReflectTiling(unittest.TestCase):

    def test_generate(self):
        for combi in COMBIS:
            tiling = ReflectTiling(combi[0], combi[1], n=MAXLAYERS)
            # this is basically the only test we can do and it only will scream when duplicates are found
            with PrintTest() as stream:
                tiling.check_integrity()

    def test_find(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi)

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
        for combi in COMBIS:
            tiling = ReflectTiling(*combi)
            for index in range(tiling.length):
                neighbors = tiling.get_neighbors(index)

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

    def test_check_integrity(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi)

            # check for duplicates
            old = tiling._sector_polys[-1, 0]
            for i in range(int(RATIOOFDUPLICATES * len(tiling._sector_polys))):
                index = random.randint(2, len(tiling._sector_polys) - 2)  # ignore last as it will be used as duplicate
                tiling._sector_polys[-1, 0] = tiling._sector_polys[index, 0]
                with self.assertRaises(AttributeError, msg=f"Duplicate at index {index} was not detected") as error:
                    tiling.check_integrity()
                self.assertEqual(f"Duplicate detected at index {index}", str(error.exception))
            tiling._sector_polys[-1, 0] = old

            # check layer completeness
            # get testing boundary for tiling
            with PrintTest() as stream:
                tiling.check_integrity()
            values = stream.get()
            messages = values[:-1].split("\n")
            if len(messages) == 1:
                boundary_layer = combi[2]
            else:
                boundary_layer = int(messages[0].split(" ")[1])

            # only the last q - 3 layers can be incomplete
            # self.assertGreaterEqual(4, 3)
            self.assertGreaterEqual(boundary_layer, combi[2] - int(np.ceil((combi[1] - 3) / 2)))

            # obscure layer l and check if layer is considered incomplete
            for l in range(boundary_layer):
                # find first polygon which belongs to the layer
                index = np.where(tiling._layers == l)
                # pretend that the polygon belongs to another layer
                tiling._layers[index] += 1
                with PrintTest() as stream:
                    tiling.check_integrity()
                tiling._layers[index] -= 1
                values = stream.get()
                messages = values[:-1].split("\n")
                # check if layer is considered incomplete
                self.assertEqual(f"Layer {l} is not complete", messages[0])

            # check layer boundary/holes
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
                self.assertIn(last_layer, [tiling._layers[index] - 1, tiling._layers[index]])


if __name__ == '__main__':
    unittest.main()
