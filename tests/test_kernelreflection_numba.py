import unittest
from tests.test_util import *
from hypertiling.array.kernelreflection_numba import ReflectTiling
import hypertiling.geodesics as geos
import hypertiling.array.reflection_numba_util as util
import random
import numpy as np

# overall
COMBIS = [(7, 3), (3, 7), (5, 4), (4, 5), (6, 4), (7, 4), (7, 5), (7, 6), (7, 7)]
LAYERS = 4

# test generate
MAXLAYERS = 15

# test check_integrity
RATIOOFDUPLICATES = 0.1
RATIOOFHOLES = 0.1

# test find
RATIOFINDS = 0.1
SHIFTTOL = 1e-8

# test get_neighbors
RATIOPOLYS = 0.1
# SHIFTTOL is also used

class TestReflectTiling(unittest.TestCase):

    def test_generate(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi, n=MAXLAYERS)
            # this is basically the only test we can do and it only will scream when duplicates are found
            with PrintTest() as stream:
                tiling.check_integrity()

    def test_find(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi, n=LAYERS)

            for i in range(int(RATIOFINDS * len(tiling.sector_polys))):
                index = random.randint(0, len(tiling.sector_polys) - 1)

                # find with center
                found_c = tiling.find(tiling.sector_polys[index, 0])
                # test boundary of findings
                neighbor_centers = util.generate_raw(tiling.sector_polys[index])
                neighbor = random.choice(neighbor_centers)
                midpoint = geos.geodesic_midpoint(neighbor, tiling.sector_polys[index, 0])
                v = (tiling.sector_polys[index, 0] - midpoint) * SHIFTTOL
                test_point = midpoint + v
                found_b = tiling.find(test_point)
                # test non finding
                center = tiling.sector_polys[index, 0]
                tiling.sector_polys[index, 0] = 0 if index != 0 else 0.999999999
                found_n = tiling.find(center)
                tiling.sector_polys[index, 0] = center

                self.assertEqual(found_c, index)
                self.assertEqual(found_b, index)
                self.assertFalse(found_n)

    def test_map_layers(self):
        self.fail()

    def test_get_neighbors(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi, n=LAYERS)
            for i in range(int(RATIOPOLYS * len(tiling.sector_polys))):
                index = random.randint(0, len(tiling.sector_polys) - 1)
                neighbors = tiling.get_neighbors(index)

                for j, poly in enumerate(tiling.sector_polys):
                    if j == index:
                        continue

                    midpoint = geos.geodesic_midpoint(tiling.sector_polys[index, 0], poly[0])
                    v = (tiling.sector_polys[index, 0] - midpoint) * SHIFTTOL
                    testpoint = midpoint + v
                    found = tiling.find(testpoint)
                    print(util.f_dist(testpoint, tiling.sector_polys[index, 0]),
                          util.f_dist(testpoint, poly[0]))

                    if j in neighbors:
                        self.assertEqual(index, found)
                    elif found is not False:
                        self.assertNotEqual(index, found)

    def test_check_integrity(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi, n=LAYERS)

            # check for duplicates
            old = tiling.sector_polys[-1, 0]
            for i in range(int(RATIOOFDUPLICATES * len(tiling.sector_polys))):
                index = random.randint(2, len(tiling.sector_polys) - 2)
                tiling.sector_polys[-1, 0] = tiling.sector_polys[index, 0]
                with self.assertRaises(AttributeError, msg=f"Duplicate at index {index} was not detected") as error:
                    tiling.check_integrity()
                self.assertEqual(f"Duplicate detected at index {index}", str(error.exception))
            tiling.sector_polys[-1, 0] = old

            # check layer completeness
            # get testing boundary for tiling
            with PrintTest() as stream:
                tiling.check_integrity()
            values = stream.get()
            messages = values[:-1].split("\n")
            if len(messages) == 1:
                boundary_layer = LAYERS
            else:
                boundary_layer = int(messages[0].split(" ")[1])

            # obscure layer l and check if layer is considered incomplete
            for l in range(boundary_layer):
                index = np.where(tiling.layers == l)
                tiling.layers[index] += 1
                with PrintTest() as stream:
                    tiling.check_integrity()
                tiling.layers[index] -= 1
                values = stream.get()
                messages = values[:-1].split("\n")
                self.assertEqual(f"Layer {l} is not complete", messages[0])

            # check layer boundary/holes
            indices = np.argwhere(tiling.layers < boundary_layer)[1:]  # eliminate fist polygon
            for i in range(int(RATIOOFHOLES * len(tiling.sector_polys))):
                index = random.choice(indices)[0]
                old = tiling.sector_polys[index, 0]
                tiling.sector_polys[index, 0] = 0
                with PrintTest() as stream:
                    tiling.check_integrity()
                tiling.sector_polys[index, 0] = old
                values = stream.get()
                messages = values[:-1].split("\n")
                last_layer = int(messages[-1].split(" ")[-1])
                self.assertIn(last_layer, [tiling.layers[index] - 1, tiling.layers[index]])


if __name__ == '__main__':
    unittest.main()
