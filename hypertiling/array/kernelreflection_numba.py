from typing import Callable
import numpy as np
import hypertiling.array.reflection_numba_util as util
from hypertiling.array.reflection_numba_util import PI2
import hypertiling.arraytransformation as trans

"""
p: Number of edges/vertices of a polygon
q: Number of polygons that meet at a vertex
n: Number of layers (classical definition)
m: Number of polygons
"""

# Magic number: real irrational number \Gamma(\frac{1}{4})
MANGLE = 3.6256099082219083119306851558676720029951676828800654674333779995


class ReflectTiling:
    """
    Creates the hyperbolic tiling.
    """

    def __init__(self, p: int, q: int, n: int, degtol: int = 0, mangle: float = MANGLE):
        """
        Initialize a hyperbolic tiling. CELL CENTERED ONLY!
        Time-complexity: O(n) + O(p^3 m(p, q, n)), with m(p, q, n) is the number of polygons
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
        self.geo_atts = (p, q, n)

        # technical attributes
        if n > 1:
            lengths = util.get_ns(self.geo_atts)
            self._sector_lengths = np.ceil(lengths / p).astype(np.uint32)
            self.length = np.sum(lengths)
        else:
            self._sector_lengths = np.array([1])
            self.length = 1

        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))
        self.degtol = degtol
        self.mangle = mangle / 360 * PI2

        # if center is added it should be p+1
        self._sector_polys = np.empty((np.sum(self._sector_lengths), p + 1), dtype=np.complex)

        """
        edge_array is not the most compact representation of the edges. The idea is to store which edges are blocked
        within a number in the array. Each polygon has its own number where the index is equal in edge_array and 
        the tiling
        
        This is saved for the possibility to expand the grid later (not yet implemented)
        """
        self._edge_array = np.empty(self._sector_polys.shape[0], dtype=np.min_scalar_type(2 ** self.geo_atts[0] - 1))

        rf = self.generate()
        self._reflection_levels = np.array([np.count_nonzero(rf == i) for i in range(np.max(rf) + 1)], dtype=np.uint32)
        self._reflection_levels_cumulated = np.empty((self._reflection_levels.shape[0] + 1,), dtype=np.uint32)
        self._reflection_levels_cumulated[0] = 0
        for i, element in enumerate(self._reflection_levels):
            self._reflection_levels_cumulated[i + 1] = element + self._reflection_levels_cumulated[i]

        # possible to fill
        self._layers = None

    def generate(self):
        """
        Calculate the tilings polygons for an angular sector.
        Time-complexity: O(p^3 m(p, q, n)), with m(p, q, n) is the number of polygons
        :return: void
        """
        return util.generate(self.geo_atts, self.r, self._sector_polys, self._sector_lengths, self._edge_array,
                             self.degtol,
                             self.mangle)

    def __len__(self):
        """
        Return the number of polygons in the tiling
        Time-complexity: O(1)
        :return: int = number of polygons in the tiling
        """
        return self.length

    def __iter__(self):
        """
        Iterates over the whole grid. As only one sector is stored in the memory, the others are generated when needed.
        Time-complexity (single polygon): O(p)
        :yield: np.array[8] = [center, vertices]
        """
        # check for duplicates
        for poly in self._sector_polys:
            yield poly

        dphi = PI2 / self.geo_atts[0]
        phis = np.array([dphi * i for i in range(1, self.geo_atts[0])])
        for i, angle in enumerate(phis):
            for poly in self._sector_polys[1:]:
                yield poly * np.exp(angle * 1j)

    def __getitem__(self, index: int) -> np.array:
        """
        Returns the center and vertices of the polygon at index. As only one sector is stored,
        the corresponding polygon is calculated if necessary.
        Time-complexity: O(p)
        :param index: int = index of the polygon
        :return: np.array[p + 1] = [center, vertices]
        """
        if index == 0:
            return self._sector_polys[0]

        # remove the first one from consideration
        index -= 1

        phi = PI2 / self.geo_atts[0] * (index // (self._sector_polys.shape[0] - 1))
        index = index if index < (self._sector_polys.shape[0] - 1) else index % (self._sector_polys.shape[0] - 1)

        # +1 to ignore the first one
        poly = self._sector_polys[index + 1]

        if phi == 0:
            return poly

        return poly * np.exp(phi * 1j)

    def _get_reflection_level_in_sector(self, index: int) -> int:
        """
        Returns the reflection level the polygon at index belongs to.
        Time-complexity: O(log(n + 1))
        :param index: int = index of the polygon
        :return: int = reflection level
        """
        pos = np.searchsorted(self._reflection_levels_cumulated, index)
        if self._reflection_levels_cumulated[pos] > index:
            return pos - 1
        return pos

    def get_layer(self, index: int) -> int:
        """
        Returns the layer, the polygon at index refers to.
        Time-complexity (with mapping): O(p^3)
        Time-complexity (without map.): O(1)
        :param index: int = index of the polygon
        :return: int = number of the layer
        """
        if self._layers is None:
            print("Layers are not yet mapped. Start mapping")
            self.map_layers()

        if index == 0:
            return self._layers[0]

        # remove the first one from consideration
        index -= 1
        index = index if index < (self._sector_polys.shape[0] - 1) else index % (self._sector_polys.shape[0] - 1)
        return self._layers[index + 1]

    def get_sector(self, index: int) -> int:
        """
        Returns the sector, the polygon at index refers to.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: int = number of the sector
        """
        if index == 0:
            return 0
        else:
            index -= 1
            return index // (self._sector_polys.shape[0] - 1)

    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.complex128 = center of the polygon
        """
        return self[index][0]

    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.array[np.complex128][p] = vertices of the polygon
        """
        return self[index][1:]

    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.complex128 = center of the polygon
        """
        return np.angle(self[index][0])

    def _find(self, sector_proj) -> int:
        """
        Protected(!)
        Find the polygons index sector_projection belongs to.
        However, sector_projection has to be in the fundamental sector.
        Time-complexity: O(m)
        :param sector_proj: complex = position to search polygon for
        :return: int = index of the corresponding polygon
        """
        disk_distance = np.vectorize(lambda z: util.f_dist(z, sector_proj))
        dists = disk_distance(self._sector_polys[:, 0])
        index = np.argmin(dists)

        if dists[index] < util.f_dist(self._sector_polys[0, 0], self._sector_polys[1, 0]) / 2:
            return index
        return False

    def find(self, v: np.complex128) -> int:
        """
        Find the polygons index v belongs to.
        Time-complexity: O(m)
        :param v: complex = position to search polygon for
        :return: int = index of the corresponding polygon
        """
        angle = np.angle(v)
        factor = int(np.floor((angle - self.degtol / 360 * PI2) / (PI2 / self.geo_atts[0])))

        for modify in [0, 1, -1]:
            modi = factor + modify
            modi = modi if modi >= 0 else modi + self.geo_atts[0]
            sector_proj = v * np.exp(-(modi * PI2 / self.geo_atts[0]) * 1j) if modi != 0 else v
            index = self._find(sector_proj)
            if index:
                index = int(index + (self._sector_polys.shape[0] - 1) * modi)
                return (index + self.length) % self.length
            elif not (index is False):
                return 0

        return False

    def map_layers(self):
        """
        This function is numerically expensive!
        Calculates the layer to each polygon.
        Time-complexity: O(m p^3)
        :return: void
        """
        verticess = [[] for i in range(self.geo_atts[2] + 2)]
        verticess[0] = self._sector_polys[0, 1:]
        self._layers = np.empty(self._sector_polys.shape[0], dtype=np.uint8)
        self._layers.fill(self.geo_atts[2])
        self._layers[0] = 0

        for i, poly in enumerate(self._sector_polys[1:], start=1):
            blocked = []
            for j, vertices in enumerate(verticess):
                if len(vertices) == 0:
                    break

                conn = util.any_close_matrix(np.array(vertices), poly[1:])
                if conn.shape[0] != 0 and len(blocked) == 0:
                    self._layers[i] = j + 1
                    if self._layers[i - 1] != 0 and self._layers[i] != self._layers[i - 1]:
                        # last vertex for layers[i - 1] set
                        # replicate vertices of sector into next one to prevent boundary problems
                        for k in range(len(verticess[self._layers[i - 1]])):
                            replicate = verticess[self._layers[i - 1]][k] * np.exp(PI2 / self.geo_atts[0] * 1j)
                            verticess[self._layers[i - 1]].append(replicate)

                blocked += conn[:, 0].tolist()

            for k, vertex in enumerate(poly[1:]):
                if k in blocked:
                    continue
                else:
                    verticess[self._layers[i]].append(vertex)

    def _polygen(self, polys: np.array) -> np.array:
        """
        Protected(!)
        Generator for iterating over polygons.
        Time-complexity (single polygon): O(p)
        :param polys: np.array[n, 8] = segment the generator will create the rotations duplicates for and rotate over
        :yield: np.array[8] = polygon of the segment polys or its rotational duplicates
        """
        for poly in polys:
            yield poly

        dphi = PI2 / self.geo_atts[0]
        phis = np.array([dphi * i for i in range(1, self.geo_atts[0])])
        for i, angle in enumerate(phis):
            for poly in polys:
                yield poly * np.exp(angle * 1j)

    def get_neighbors(self, index: int) -> np.array:
        """
        Get the neighbors of a polygon at index
        Time-complexity: O(m)
        :param index: int = index of the polygon
        :return: np.array[p] = array containing the indices of the neighbors
        """
        if index == 0:
            neigbor_centers = util.generate_raw(self._sector_polys[index])
            indices = [self.find(e) for e in neigbor_centers]
            indices = [e for e in indices if not (e is False)]
            return [i % (self.length - 1) if i != 0 else 0 for i in indices]

        # get equivalent poly in sector
        index -= 1
        sector_replica = index // (self._sector_polys.shape[0] - 1)
        index %= (self._sector_polys.shape[0] - 1)
        index += 1
        jump = self._sector_polys.shape[0] - 1

        # quick and dirty solution
        neighbor_centers = util.generate_raw(self._sector_polys[index])
        indices = [self.find(e) for e in neighbor_centers]
        indices = [e for e in indices if not (e is False)]
        indices = [(i + sector_replica * jump) if i != 0 else 0 for i in indices]
        return [i if i < self.length else i % self.length + 1 for i in indices]

    def get_neighbors_fast(self, index: int) -> np.array:
        # FIXME: at first for sector... Rotate later
        if index == 0:
            return np.array([1 + i * (self._sector_polys.shape[0] - 1) for i in range(self.geo_atts[0])])

        # placeholder for the neighbors
        neighbors = np.empty((self.geo_atts[0]), dtype=np.uint32)

        # map index to sector
        sector_index = index - 1
        sector_index = sector_index + 1 if sector_index < (self._sector_polys.shape[0] - 1) else sector_index % (
                self._sector_polys.shape[0] - 1) + 1

        c = 0
        jump = (self._sector_polys.shape[0] - 1)

        layer = self._get_reflection_level_in_sector(sector_index)
        pos_in_layer = sector_index - self._reflection_levels_cumulated[layer]
        ratio = pos_in_layer / self._reflection_levels[layer]

        # siblings
        # For q == 3, the polys have direct contact to their siblings. Otherwise q - 3 elements are between them
        if self.geo_atts[1] == 3:
            neighbors[c] = sector_index - 1 if sector_index - 1 >= self._reflection_levels_cumulated[layer] else \
                self._reflection_levels_cumulated[layer] + self._reflection_levels[layer] + (
                        self.geo_atts[0] - 1) * jump - 1

            neighbors[c + 1] = sector_index + 1 \
                if sector_index <= self._reflection_levels[layer] else \
                self._reflection_levels_cumulated[layer] + jump

            c += 2
        #  For q is odd follows (q - 3) is even. If (q - 3) is even, a polygon can connect exact one other sibling
        pass

        # parent
        neighbors[c] = self._reflection_levels_cumulated[layer - 1] + int(ratio * self._reflection_levels[layer - 1])
        c += 1

        print(bin(self._edge_array[c]))
        print(bin(1 << (self.geo_atts[0] - 1)))
        if not (self._edge_array[c] & 1 << (self.geo_atts[0] - 1)):
            neighbors[c] = self._reflection_levels_cumulated[layer - 1] + int(
                ratio * self._reflection_levels[layer - 1]) + 1
            if neighbors[c] >= self._reflection_levels_cumulated[layer]:
                neighbors[c] = self._reflection_levels_cumulated[layer - 1] + jump
            c += 1
        """
        Kann einen oder zwei parents geben
        2 nur, wenn es zwischen diesen liegt. Wenn es das tut, dann ist in edge_array die letzte ecke gesperrt
        """

        # children
        if layer + 1 == self.geo_atts[2]:
            return neighbors[:c]

        neighbors[c] = self._reflection_levels_cumulated[layer + 1] + int(ratio * self._reflection_levels[layer + 1])
        c += 1

        up = 1
        while c < self.geo_atts[0]:
            neighbors[c] = self._reflection_levels_cumulated[layer + 1] + int(
                ratio * self._reflection_levels[layer + 1]) + up
            if neighbors[c] < self._reflection_levels_cumulated[layer + 1]:
                neighbors[c] = self._reflection_levels_cumulated[layer + 1] + self._reflection_levels[layer + 1] + (
                        self.geo_atts[0] - 1) * jump - 1
            elif neighbors[c] >= self._reflection_levels_cumulated[layer + 2]:
                neighbors[c] = self._reflection_levels_cumulated[layer] + jump
            c += 1
            up = - up if np.sign(up) == 1 else - up + 1

        return neighbors

    def check_integrity(self):
        """
        This function is numerically expensive!
        Checks the integrity of the grid. The number of neighbors as well as a search for duplicates is applied.
        Raises AttributeError if the grid seems to be invalid.
        Time-complexity: O(m p^3 n)
        :return: void
        """
        # check if one polygon is shifted in the range of another or if duplicates exist
        for i in range(len(self._sector_polys)):
            poly_center = self._sector_polys[i, 0]
            self._sector_polys[i, 0] = 0
            try:
                if self.find(poly_center):
                    raise AttributeError(f"Duplicate detected at index {i}")
            finally:
                self._sector_polys[i, 0] = poly_center

        # check if each layer has the correct size
        if self._layers is None:
            self.map_layers()

        for i, length in enumerate(self._sector_lengths):
            if np.count_nonzero(self._layers == i) != length:
                print(f"Layer {i} is not complete")
                break

        # check if all edges have a partner
        for i in range(len(self._sector_polys)):
            neighbor_counter = len(self.get_neighbors(i))
            if neighbor_counter == self.geo_atts[0]:
                continue
            print(f"Integrity ensured till index {i} at layer {self.get_layer(i)}")
            return

    def transform(self, function: Callable):
        """
        Applies function to each polygon
        :param function: callable = function to apply on each polygon
        Time-complexity: O(m)
        :return: void
        """
        if not isinstance(function, np.vectorize):
            function = np.vectorize(function)
        function(self._sector_polys)

    def rotate(self, angle: float):
        """
        Rotates the grid around angle
        :param angle: float = angle to rotate the polygon
        Time-complexity: O(m)
        :return: void
        """
        self.transform(lambda x: trans.mrotate(x.shape[0], -angle, x))

    def translate(self, z: np.complex128):
        """
        Translates the grid to z
        :param z: complex = position of the new origin
        Time-complexity: O(m)
        :return: void
        """
        self.transform(lambda x: trans.morigin(x.shape[0], z, x))


if __name__ == "__main__":
    import hypertiling.array.plot as plot
    import matplotlib.pyplot as plt
    import time
    import random
    import matplotlib as mpl

    tiling = ReflectTiling(7, 3, 2)

    p, q, = 3, 7
    print("\n\n")
    t1 = time.time()
    tiling = ReflectTiling(p, q, 4)
    print(f"{time.time() - t1} s")

    # plot.plot(tiling, alpha=0.5)
    # tiling.map_layers()
    fig, ax = plt.subplots()

    colors = [(random.random(), random.random(), random.random(), 0.5) for i in range(5)]
    for i, pgon in enumerate(tiling):
        pgon = tiling[i]
        center = tiling.get_vertices(i)
        plt.scatter(np.real(center), np.imag(center))
        p_ = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), lw=1, edgecolor="#FFFFFF",
                                 fc=colors[tiling.get_layer(i)])
        ax.add_patch(p_)
        ax.text(np.real(pgon[0]), np.imag(pgon[0]), i, horizontalalignment='center', verticalalignment='center')

    tiling.check_integrity()
    plt.title(f"{p} {q}")

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    arrow = np.array([1, 0])
    # lower boundary
    plt.arrow(0, 0, arrow[0], arrow[1])

    # upper boundary
    theta = PI2 / tiling.geo_atts[0] + (tiling.degtol / 360 * PI2)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    arrow = rot @ arrow
    plt.arrow(0, 0, arrow[0], arrow[1])

    plt.show()

"""
Arbeitsplan:
    - neighbors fast
    - speed tests
"""
