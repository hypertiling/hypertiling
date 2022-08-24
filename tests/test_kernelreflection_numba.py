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

# test find/get_neighbors
SHIFTTOL = 1e-8


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

            for index in range(tiling.length - 1):
                center = tiling[index][0]

                # find with center
                self.assertEqual(tiling.find(center), index)

                # test boundary of findings
                """neighbor_centers = util.generate_raw(tiling[index])
                for neighbor in neighbor_centers:
                    midpoint = geos.geodesic_midpoint(neighbor, center)
                    v = (center - midpoint) * SHIFTTOL
                    test_point = midpoint + v
                    self.assertEqual(tiling.find(test_point), index)"""

                # test non finding
                if 1 < index < len(tiling.sector_polys): # 0 and 1 are excluded as they are used as reference in find
                    tiling.sector_polys[index, 0] = 0 if index != 0 else 0.999999999999j
                    self.assertFalse(tiling.find(center))
                    tiling.sector_polys[index, 0] = center

    def test_map_layers(self):
        self.fail()

    def test_get_neighbors(self):
        for combi in COMBIS:
            tiling = ReflectTiling(*combi, n=LAYERS)
            dist = util.f_dist(tiling[1][0], tiling[0][0]) / 2
            for index in range(tiling.length):
                neighbors = tiling.get_neighbors(index)

                for j, poly in enumerate(tiling.sector_polys):
                    if j == index:
                        continue

                    midpoint = geos.geodesic_midpoint(tiling[index][0], poly[0])
                    v = (tiling[index][0] - midpoint) * 0.5#SHIFTTOL
                    testpoint = midpoint + v
                    found = tiling.find(testpoint)

                    if j in neighbors:
                        try:
                            self.assertEqual(index, found)
                        except Exception as error:
                            import hypertiling.array.plot as plot
                            import matplotlib.pyplot as plt
                            print("neighbor")
                            print(testpoint)
                            print(index, j, found, neighbors)
                            print(util.f_dist(tiling[index][0], testpoint))
                            print(dist)

                            neighbors_ = np.array([tiling[k][0] for k in neighbors])
                            plot.plot(tiling, numerate=True, alpha=0.5)
                            plt.scatter(np.real(testpoint), np.imag(testpoint), color="#FF0000", marker="x")
                            plt.scatter(np.real(midpoint), np.imag(midpoint))
                            plt.scatter(np.real(neighbors_), np.imag(neighbors_))
                            plt.show()
                            raise error

                    """elif found is not False:
                        try:
                            self.assertNotEqual(index, found)
                        except Exception as error:
                            import hypertiling.array.plot as plot
                            import matplotlib.pyplot as plt
                            print(index, found)
                            print(index, j, neighbors)
                            neighbors_ = np.array([tiling[k][0] for k in neighbors])
                            plot.plot(tiling, numerate=True, alpha=0.5)
                            plt.scatter(np.real(neighbors_), np.imag(neighbors_))
                            plt.show()
                            raise error"""

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
