from typing import Callable, Any, List
import numpy as np
import hypertiling.generative.generative_reflection_util as util
from hypertiling.generative.generative_reflection_util import PI2
import hypertiling.graph.generative_reflection_graph_util as graph_util

"""
p: Number of edges/vertices of a polygon
q: Number of polygons that meet at a vertex
n: Number of layers (reflective definition)
m: Number of polygons
m = m(p, q, n)

LIMITATIONS:
- A reflection layer can hold at max 4294967295 polys as the size is stored as uint32 (util.get_reflection_n_estimation)
- The whole tiling can holy at max 34359738353 polys as the size of _sector_polys is determined as sum of uint32 of the 
  layers size in the fundamental sector
- The number of reflection layers is limited to 255 at max, as util.generate stores the layers as uint8
"""

# Magic number: real irrational number \Gamma(\frac{1}{4})
MANGLE = 3.6256099082219083119306851558676720029951676828800654674333779995


class KernelGenerativeReflectionGraph:
    """
    Creates the hyperbolic tiling.
    """

    def __init__(self, p: int, q: int, n: int, degtol: int = 0, mangle: float = MANGLE):
        """
        Initialize a hyperbolic tiling. CELL CENTERED ONLY!
        Time-complexity: O(p^2 m + n + m / p * n)
        :param p: int = number of vertices per cells
        :param q: int = number of cells meeting at each vertex
        :param n: int =  number of layers to be constructed
        :param degtol: int = tolerance at boundary in degrees
        :param mangle: float = rotation of the center polygon in degrees
                               (prevents boundaries from being along symmetry axis)
        """

        # grid attributes
        if not ((p - 2) * (q - 2) > 4):
            raise AttributeError("Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

        self.p = p
        self.q = q
        self.n = n

        # technical attributes
        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))
        self.degtol = degtol
        self.mangle = mangle / 360 * PI2

        # estimate some other technical attributes
        if n != 0:
            lengths = util.get_reflection_n_estimation(p, q, n)  # n
            self._sector_lengths = lengths # np.ceil(lengths / p).astype(np.uint32)  # n
        else:
            self._sector_lengths = np.array([1])

        self.graph, self.center_coords = self._generate()
        self.length = (self.graph.shape[0] - 1) * self.p + 1

    def __getitem__(self, item):
        """
        Get neighbor of the polygon at index.
        Time-complexity (single polygon): O(p)
        :param item: int = index of the polygon for whom the neighbors will be searched for
        :return: np.array = indices of the neighbors
        """
        return self._expand_sector_index_to_tiling(item, self._get_nbrs)

    def _generate(self):
        return graph_util.generate_nbrs(self.p, self.q, self.n, self.r, self._sector_lengths, self.degtol, self.mangle)

    @staticmethod
    def _to_weierstrass(polygons: np.array) -> np.array:
        """
        Protected(!)
        Calculates the weierstrass coordinates for an array of polygons in poincare disks.
        Time-complexity: O(m / p)
        :param polygons: np.array[n, p + 1] = polygons to calculate weierstrass coordinates for
        :return: np.array[p + 1, 3] = polygons in weierstrass coordinates
        """
        weierstrass = np.empty((len(polygons), 3), dtype=np.float64)
        weierstrass[:, 0] = 1
        weierstrass[:, 1] = np.real(polygons[:, 0])
        weierstrass[:, 2] = np.imag(polygons[:, 0])
        xx, yy = weierstrass[:, 1] * weierstrass[:, 1], weierstrass[:, 2] * weierstrass[:, 2]
        weierstrass[:, 0] += xx + yy
        weierstrass /= (1 - (xx + yy))[:, None]
        weierstrass[:, 1] *= 2
        weierstrass[:, 2] *= 2
        return weierstrass

    def _expand_sector_index_to_tiling(self, index: int, f: Callable) -> Any:
        """
        Protected(!)
        Takes an index (for the tiling) and a function defined in the fundamental sector.
        Calculates the corresponding sector_index, applies function f, and corrects the result to index.
        Time-complexity: O(f(index))
        :param index: int = index of a polygon in the tiling
        :param f: Callable = function to apply on sector_index
        :return: np.array[p + 1] = polygon of the segment polys or its rotational duplicates
        """
        if index != 0:
            # get equivalent poly in sector
            index -= 1
            sector_replica = index // (self.graph.shape[0] - 1)
            index %= (self.graph.shape[0] - 1)
            index += 1
            jump = self.graph.shape[0] - 1

            indices = f(index)

            indices = [(i + sector_replica * jump) if i != 0 else 0 for i in indices]
            return [i if i < self.length else i % self.length + 1 for i in indices]

        return f(index)

    def _get_nbrs(self, sector_index: int) -> np.array:
        """
        Protected(!)
        Get neighbor of the polygon at sector_index. Has to be in the fundamental sector!
        Time-complexity (without map.): O(p)
        :param sector_index: int = index of the polygon for whom the neighbors will be searched for
        :return: np.array = indices of the neighbors
        """
        neighbor_indices = self.graph[sector_index]

        # get value from nice little overflow
        overflow = np.iinfo(neighbor_indices.dtype).max
        return neighbor_indices[np.argwhere(neighbor_indices != overflow)].flatten()  # p

    def check_integrity(self):
        print("TODO: abstände zwischen den Nachbarn kontrollieren!")  # TODO
        neighbors = self.get_nbrs_list()
        for i, nbrs in enumerate(neighbors):
            if len(nbrs) != self.p:
                print(f"Integrity ensured till index {i}. {i} has only {len(nbrs)} neighbors")
                break

    def get_nbrs_list_sector(self) -> List[List[int]]:
        max_number = np.iinfo(self.graph.dtype).max
        return [[index for index in row if index != max_number] for row in self.graph.tolist()]

    def get_nbrs_list(self) -> List[List[int]]:
        """
        Create and return list of all neighbors
        Time-complexity: O(mp)
        :param tol: float = tolerance to search neighbors in
        :return: List[List[int]] = list of all neighbors for all polygons
        """
        part = np.copy(self.graph[1:]).astype(np.uint32)  # m / p * p = m
        max_number = np.iinfo(self.graph.dtype).max  # m / p

        jump = np.uint32(self.graph.shape[0] - 1)
        rotate = np.vectorize(lambda x: x if x == max_number else x if x == 0 else x + jump)
        neighbors = [[element for element in line if element != max_number] for line in self.graph.tolist()]
        # m / p loop execs: p loop execs: O(1)

        for sector_i in range(1, self.p):  # p loop execs
            part = rotate(part)  # m / p * p = m
            neighbors += [[i if i < self.length else i % self.length + 1 for i in line if i != max_number] for line in
                          part.tolist()]
            # m / p loop execs: p loop execs: O(1)

        return neighbors


if __name__ == "__main__":
    import time
    from hypertiling.generative.generative_reflection import KernelGenerativeReflection
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    p, q, n = 5, 4, 3
    t1 = time.time()
    graph = KernelGenerativeReflectionGraph(p, q, n)
    print(f"Took: {time.time() - t1}")

    """t1 = time.time()
    tiling = KernelGenerativeReflection(p, q, n)
    print(f"Took: {time.time() - t1}")"""

    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)
    graph.check_integrity()
    graph_util.plot_graph(graph.get_nbrs_list(), graph.center_coords, graph.p)

    """colors = ["#FF000080", "#00FF0080", "#0000FF80"]
    for polygon_index, pgon in enumerate(tiling):
        poly_layer = tiling.get_reflection_level(polygon_index)
        facecolor = colors[poly_layer % len(colors)]
        patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]),
                                    facecolor=facecolor, edgecolor="#FFFFFF")
        fig_ax[1].add_patch(patch)
        fig_ax[1].text(np.real(pgon[0]), np.imag(pgon[0]), str(polygon_index))"""
    plt.show()


