from typing import Callable, Any, List
import numpy as np
import hypertiling.kernel.GR_util as util
import hypertiling.kernel.GRG_util as graph_util
from hypertiling.kernel_abc import GraphExtended
from hypertiling.ion import htprint

# Magic number: real irrational number \Gamma(\frac{1}{4})
MANGLE = np.radians(3.6256099082219083119306851558676720029951676828800654674333779995)


class GenerativeReflectionGraph(GraphExtended):
    """
    Following the same algorithmic principles as the GenerativeReflection (GR) kernel, this class constructs 
    neighborhood relations already during the construction of the lattice. Only one symmetry sector 
    is explicitly stored, whereas any information outside this sector is generated on demand. 
    Geometric cell information, except for the center coordinates, is not stored.

    p: Number of edges/vertices of a polygon
    q: Number of polygons that meet at a vertex
    n: Number of layers (reflective definition)
    m: Number of polygons
    m = m(p, q, n)

    LIMITATIONS:
    - A reflection layer can hold at max 4.294.967.295 polys as the size is stored as uint32 (util.get_reflection_n_estimation)
    - The whole tiling can hold at max 34.359.738.353 polys as the size of _sector_polys is determined as sum of uint32 of the 
    layers size in the fundamental sector
    - The number of reflection layers is limited to 255 at max, as util.generate stores the layers as uint8
    """

    def __init__(self, p: int, q: int, n: int, tol: float = 1e-8, mangle: float = MANGLE):
        """
        Initialize a hyperbolic tiling. CELL CENTERED ONLY!

        Time-complexity: O(p^2 m + n + m / p * n)

        Parameters
        ----------
        p : int
            Number of vertices per cells.
        q : int
            Number of cells meeting at each vertex.
        n : int
            Number of layers to be constructed.
        degtol : int
            Tolerance at boundary in degrees.
        tol : int
            Tolerance at boundary for neighbor matching.
        mangle : float
            Rotation of the center polygon in degrees (prevents boundaries from being along symmetry axis).
        """
        super().__init__(p, q, n, mangle)

        # technical attributes
        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))
        self.tol = tol

        # estimate some other technical attributes
        if n != 0:
            lengths = util.get_reflection_n_estimation(p, q, n)  # n
            self._sector_lengths = np.ceil(lengths / p).astype(np.uint32)  # n
        else:
            self._sector_lengths = np.array([1])

        self._nbrs, self.center_coords = self._generate()
        self.length = (self._nbrs.shape[0] - 1) * self.p + 1

        self._sector_lengths_cumulated = np.empty((self._sector_lengths.shape[0] + 1,), dtype=np.uint32)
        self._sector_lengths_cumulated[0] = 0
        for i, element in enumerate(self._sector_lengths):  # n loop execs
            self._sector_lengths_cumulated[i + 1] = element + self._sector_lengths_cumulated[i]

    def __getitem__(self, item):
        """
        Get neighbor of the polygon at index.

        Time-complexity (single polygon): O(p)

        Parameters
        ----------
        item : int
            Index of the polygon for whom the neighbors will be searched for.

        Returns
        -------
        np.array
            Indices of the neighbors.
        """
        return self.get_nbrs(item)

    def __len__(self):
        """
        Return the number of polygons in the tiling

        Time-complexity: O(1)

        Returns
        -------
        int
            Number of polygons in the tiling
        """
        return self.length

    # Helper ###########################################################################################################

    def _generate(self):
        """
        Generates the graph structure for the specified tiling.

        Returns
        -------
        np.array[uint32, uint32]
            Array containing the neighbor relations.
        """
        return graph_util.generate_nbrs(self.p, self.q, self.r, self._sector_lengths, self.mangle,
                                        self.tol)

    def _expand_sector_index_to_tiling(self, index: int, f: Callable) -> Any:
        """
        Protected(!)
        Takes an index (for the tiling) and a function defined in the fundamental sector.
        Calculates the corresponding sector_index, applies function f, and corrects the result to index.

        Time-complexity: O(f(index))

        Parameters
        ----------
        index : int
            Index of a polygon in the tiling.
        f : Callable
            Function to apply on sector_index.

        Returns
        -------
        np.array[p + 1]
            Polygon of the segment polys or its rotational duplicates.
        """
        if index != 0:
            # get equivalent poly in sector
            index -= 1
            sector_replica = index // (self._nbrs.shape[0] - 1)
            index %= (self._nbrs.shape[0] - 1)
            index += 1
            jump = self._nbrs.shape[0] - 1

            indices = f(index)

            indices = [(i + sector_replica * jump) if i != 0 else 0 for i in indices]
            return [i if i < self.length else i % self.length + 1 for i in indices]

        return f(index)

    def _get_nbrs(self, sector_index: int) -> np.array:
        """
        Protected(!)
        Get neighbor of the polygon at sector_index. Has to be in the fundamental sector!

        Time-complexity: O(p)

        Parameters
        ----------
        sector_index : int
            Index of the polygon for whom the neighbors will be searched for.

        Returns
        -------
        np.array
            Indices of the neighbors.
        """
        neighbor_indices = self._nbrs[sector_index]

        # get value from nice little overflow
        overflow = np.iinfo(neighbor_indices.dtype).max
        return neighbor_indices[np.argwhere(neighbor_indices != overflow)].flatten()  # p

    def _get_reflection_level_in_sector(self, sector_index: int) -> int:
        """
        Protected(!)
        Returns the reflection level the polygon at index belongs to.

        Time-complexity: O(log(n + 1))

        Parameters
        ----------
        sector_index : int
            Index of the polygon.

        Returns
        -------
        int
            Reflection level.
        """
        pos = np.searchsorted(self._sector_lengths_cumulated, sector_index)
        if self._sector_lengths_cumulated[pos] > sector_index:
            return pos - 1
        return pos

    # Helper ###########################################################################################################

    def get_coord(self, index: int) -> np.complex128:
        """
        Get the coordinates for the center of the node at index.

        Time-complexity: O(1)

        Parameters
        ----------
        index : int
            Index of the node of consideration.

        Returns
        -------
        np.complex128
            Center of the node in complex coordinates.
        """
        if index >= self.center_coords.shape[0]:
            sector = (index - 1) // (self.center_coords.shape[0] - 1)
            index = (index - 1) % (self.center_coords.shape[0] - 1)
            index += 1
            return self.center_coords[index] * np.exp(1j * sector * np.pi * 2 / self.p)
        else:
            return self.center_coords[index]

    def check_integrity(self, tol: float = 1e-8):
        """
        Controls the integrity of the tiling. Checks for correct number of neighbors and their distances.

        Time-complexity: O(mp)

        Parameters
        ----------
        tol : float
            Tolerance of the neighboring distance (to accept).

        Returns
        -------
        void
        """
        dist_ref = util.f_dist_disc(self.get_coord(0), self.get_coord(1)) + tol  # 1
        for i in range(self.length):  # exec loop m times
            nbrs = self[i]  # 1
            coord = self.get_coord(i)  # 1
            if len(nbrs) != self.p:
                print(f"Integrity ensured till index {i}. {i} has only {len(nbrs)} neighbors")
                break
            for nbr in nbrs:  # exec loop p times
                if dist_ref < util.f_dist_disc(coord, self.get_coord(nbr)):  # 1
                    raise AttributeError(f"[hypertiling] Error: Neighbor {nbr} of polygon {i} out of reach!")

    def get_nbrs_list_sector(self) -> List[List[int]]:
        """
        Returns a list of lists of the neighbors for the graph.

        Time-complexity: O(m)

        Returns
        -------
        List[List[int]]
            List for each polygon's neighbors.
        """
        if len(self) == 1:
            htprint("Warning", "Tiling consists of one polygon!")
            return []
        max_number = np.iinfo(self._nbrs.dtype).max
        return [[index for index in row if index != max_number] for row in self._nbrs.tolist()]

    def get_nbrs_list(self) -> List[List[int]]:
        """
        Create and return list of all neighbors

        Time-complexity: O(mp)

        Returns
        -------
        List[List[int]]
            List of all neighbors for all polygons.
        """
        if len(self) == 1:
            htprint("Warning", "Tiling consists of one polygon!")
            return []

        part = np.copy(self._nbrs[1:]).astype(np.uint32)  # m / p * p = m
        max_number = np.iinfo(self._nbrs.dtype).max  # m / p

        jump = np.uint32(self._nbrs.shape[0] - 1)
        rotate = np.vectorize(lambda x: x if x == max_number else x if x == 0 else x + jump)
        neighbors = [[element for element in line if element != max_number] for line in self._nbrs.tolist()]
        # m / p loop execs: p loop execs: O(1)

        for sector_i in range(1, self.p):  # p loop execs
            part = rotate(part)  # m / p * p = m
            neighbors += [[i if i < self.length else i % self.length + 1 for i in line if i != max_number] for line in
                          part.tolist()]
            # m / p loop execs: p loop execs: O(1)

        return neighbors

    def get_nbrs(self, index: int) -> List[int]:
        """
        Create and return list of all neighbors of index

        Time-complexity: O(p)

        Returns
        -------
        List[int]
            List of all neighbors for polygon at index.
        """
        if len(self) == 1:
            htprint("Warning", "Tiling consists of one polygon!")
            return []
        return self._expand_sector_index_to_tiling(index, self._get_nbrs)

    def get_reflection_level(self, index) -> int:
        """
        Get the neighbors of a polygon at index

        Time-complexity: O(log(n + 1))

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        np.array
            Array containing the indices of the neighbors.
        """
        if index == 0:
            return 0

        index -= 1
        index %= (self._sector_lengths_cumulated[-1] - 1)
        index += 1
        return self._get_reflection_level_in_sector(index)  # log(n + 1)


if __name__ == "__main__":
    import time
    from hypertiling.kernel.GR import GenerativeReflection
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import hypertiling.core as core
    from hypertiling.kernel_abc import Tiling

    p, q, n = 7, 3, 4
    n2 = 3
    t1 = time.time()
    graph = GenerativeReflectionGraph(p, q, n)
    print(f"Took: {time.time() - t1}")

    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)
    graph.check_integrity()

    colors = ["#FF000080", "#00FF0080", "#0000FF80"]

    graph_util.plot_graph(graph.get_nbrs_list(), graph.center_coords, graph.p,
                          colors=[colors[graph.get_reflection_level(i) % len(colors)] for i in range(graph.length)])
    plt.show()

