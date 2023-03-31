import unittest
from tests.test_util import *
from hypertiling.core import HyperbolicGraph, GraphKernels
from hypertiling.kernel.GRGS import GenerativeReflectionGraphStatic
from hypertiling.kernel_abc import Graph
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
RATIOOFMISSES = 0.1
RATIOOFWRONGNEIGHBORS = 0.1


class TestReflectTiling(unittest.TestCase):

    def test_generate(self):
        for combi in Progress(combis):
            with PrintTest():
                graph = HyperbolicGraph(combi[0], combi[1], n=combi[2], kernel=GraphKernels.GenerativeReflectionGraphStatic)

            # check if the tiling is from the correct kernel
            self.assertTrue(isinstance(graph, Graph))
            self.assertTrue(isinstance(graph, GenerativeReflectionGraphStatic))
            # this is basically the only test we can do and it only will scream when duplicates are found
            with PrintTest() as stream:
                graph.check_integrity()

    def test_get_nbrs(self):
        # just a simple test but check_integrity should detect problems with neighbors already
        for combi in Progress(combis):
            with PrintTest():
                graph = HyperbolicGraph(combi[0], combi[1], n=combi[2], kernel=GraphKernels.GenerativeReflectionGraph)
            for index in range(graph.length):
                neighbors = graph.get_nbrs(index)
                rf = graph.get_reflection_level(index)
                if rf + 1 == combi[2]:
                    break

                self.assertEqual(len(neighbors), combi[0])

    def test_get_nbrs_list(self):
        for combi in Progress(combis):
            with PrintTest():
                graph = HyperbolicGraph(*combi, kernel=GraphKernels.GenerativeReflectionGraph)
            neighbors_list = graph.get_nbrs_list()
            for index in range(graph.length):
                self.assertTrue(np.array_equal(neighbors_list[index], graph[index]))

    def test_check_integrity(self):
        for combi in Progress(combis):
            with PrintTest():
                graph = HyperbolicGraph(*combi, kernel=GraphKernels.GenerativeReflectionGraph)

            # check if it will detect missing neighbors
            p_1 = combi[0] - 1
            max_index = graph._sector_lengths_cumulated[-2] - 1
            for i in range(int(RATIOOFMISSES * len(graph._nbrs))):
                index = random.randint(0, max_index)
                nbr_index = random.randint(0, p_1)
                original = graph._nbrs[index, nbr_index]
                graph._nbrs[index, nbr_index] = np.iinfo(original.dtype).max
                with PrintTest() as stream:
                    graph.check_integrity()
                graph._nbrs[index, nbr_index] = original
                self.assertEqual(stream.get(), f"Integrity ensured till index {index}. {index} has only {p_1} neighbors\n")

        # check for wrong neighbors
        for i in range(int(RATIOOFWRONGNEIGHBORS * len(graph._nbrs))):
            index = random.randint(0, max_index)
            nbrs = graph[index]

            # get another polygon which is not(!) self or a neighbor
            index2 = index
            while index2 == index or index2 in nbrs:
                index2 = random.randint(0, max_index)

            nbr_index = random.randint(0, p_1)
            original = graph._nbrs[index, nbr_index]
            graph._nbrs[index, nbr_index] = index2

            with self.assertRaises(AttributeError, msg=f"Wrong neighbor {index2} at index {index} was not detected!") as error:
                graph.check_integrity()
            self.assertEqual(
                f"[hypertiling] Error: Neighbor {index2} of polygon {index} out of reach!", str(error.exception))
            graph._nbrs[index, nbr_index] = original

    def test_get_reflection_levels(self):
        for combi in Progress(combis):
            with PrintTest():
                graph = HyperbolicGraph(combi[0], combi[1], n=combi[2], kernel=GraphKernels.GenerativeReflectionGraph)

            last_layer = 0
            for index in range(graph._nbrs.shape[0]):
                rl = graph.get_reflection_level(index)
                self.assertGreaterEqual(rl, last_layer)
                last_layer = rl
            self.assertEqual(last_layer + 1, combi[2])


if __name__ == '__main__':
    unittest.main()
