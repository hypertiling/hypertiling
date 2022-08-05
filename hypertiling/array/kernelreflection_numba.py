import numpy as np
import reflection_numba_util as util
from reflection_numba_util import PI2
import hypertiling.arraytransformation as trans

# FIXME: remove me later
import warnings

# Magic number: real irrational number \Gamma(\frac{1}{4})
MANGLE = 3.6256099082219083119306851558676720029951676828800654674333779995


class ReflectTiling:
    """
    Creates the hyperbolic tiling.
    """

    def __init__(self, p, q, n, degtol=0, mangle=MANGLE):
        """
        Initialize a hyperbolic tiling. CELL CENTERED ONLY!
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
            self.sector_lengths = np.ceil(lengths / p).astype(np.uint32)
            self.length = np.sum(lengths)
        else:
            self.sector_lengths = np.array([1])
            self.length = 1

        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))
        self.degtol = degtol
        self.mangle = mangle / 360 * PI2

        # if center is added it should be p+1
        self.sector_polys = np.empty((np.sum(self.sector_lengths), p + 1), dtype=np.complex)
        """
        edge_array is not the most compact representation of the edges. The idea is to store which edges are blocked
        within a number in the array. Each polygon has its own number where the index is equal in edge_array and 
        the tiling
        """
        self.edge_array = np.empty(self.sector_polys.shape[0], dtype=np.min_scalar_type(2 ** self.geo_atts[0] - 1))
        self.generate()

        # possible to fill
        self.layers = None

    def generate(self):
        """
        Calculate the tilings polygons for an angular sector.
        :return: void
        """
        util.generate(self.geo_atts, self.r, self.sector_polys, self.sector_lengths, self.edge_array, self.degtol,
                      self.mangle)

    def __len__(self):
        """
        Return the number of polygons in the tiling
        :return: int = number of polygons in the tiling
        """
        return self.length

    def __iter__(self):
        """
        Iterates over the whole grid. As only one sector is stored in the memory, the others are generated when needed.
        :yield: np.array[8] = [center, vertices]
        """
        # check for duplicates
        for poly in self.sector_polys:
            yield poly

        dphi = PI2 / self.geo_atts[0]
        phis = np.array([dphi * i for i in range(1, self.geo_atts[0])])
        for i, angle in enumerate(phis):
            for poly in self.sector_polys[1:]:
                yield poly * np.exp(angle * 1j)

    def __getitem__(self, index):
        """
        Returns the center and vertices of the polygon at index. As only one sector is stored,
        the corresponding polygon is calculated if necessary.
        :param index: int = index of the polygon
        :return: np.array[8] = [center, vertices]
        """
        if index == 0:
            return self.sector_polys[0]

        # remove the first one from consideration
        index -= 1

        phi = PI2 / self.geo_atts[0] * (index // (self.sector_polys.shape[0] - 1))
        index = index if index < (self.sector_polys.shape[0] - 1) else index % (self.sector_polys.shape[0] - 1)

        # +1 to ignore the first one
        poly = self.sector_polys[index + 1]
        return poly * np.exp(phi * 1j)

    def get_layer(self, index):
        """
        Returns the layer, the polygon at index refers to.
        :param index: int = index of the polygon
        :return: int = number of the layer
        """
        if self.layers is None:
            warnings.warn("Layers are not yet mapped. Start mapping")
            self.map_layers()

        if index == 0:
            return self.layers[0]

        # remove the first one from consideration
        index -= 1
        index = index if index < (self.sector_polys.shape[0] - 1) else index % (self.sector_polys.shape[0] - 1)
        return self.layers[index + 1]

    def find(self, v):
        """
        Find the polygons index v belongs to.
        :param v: complex = position to search polygon for
        :return: int = index of the corresponding polygon
        """
        angle = np.angle(v)
        factor = (angle - self.degtol / 360 * PI2) // (PI2 / self.geo_atts[0])
        factor = factor if factor >= 0 else factor + self.geo_atts[0]
        sector_proj = v * np.exp(-(factor * PI2 / self.geo_atts[0]) * 1j)

        disk_distance = np.vectorize(lambda z: util.f_dist(z, sector_proj))

        dists = disk_distance(self.sector_polys[:, 0])
        index = np.argmin(dists)
        if dists[index] >= util.f_dist(self.sector_polys[0, 0], self.sector_polys[1, 0]) / 2:
            return False
        return int(index + (self.sector_polys.shape[0] - 1) * factor if index != 0 else 0)

    def map_layers(self):
        """
        This function is numerically expensive!
        Calculates the layer to each polygon.
        :return: void
        """
        verticess = [[] for i in range(self.geo_atts[2] + 2)]
        verticess[0] = self.sector_polys[0, 1:]
        self.layers = np.empty(self.sector_polys.shape[0], dtype=np.uint8)
        self.layers.fill(self.geo_atts[2])
        self.layers[0] = 0

        for i, poly in enumerate(self.sector_polys[1:], start=1):
            blocked = []
            for j, vertices in enumerate(verticess):
                if len(vertices) == 0:
                    break

                conn = util.any_close_matrix(np.array(vertices), poly[1:])
                if conn.shape[0] != 0 and len(blocked) == 0:
                    self.layers[i] = j + 1
                    if self.layers[i - 1] != 0 and self.layers[i] != self.layers[i - 1]:
                        # last vertex for layers[i - 1] set
                        # replicate vertices of sector into next one to prevent boundary problems
                        for k in range(len(verticess[self.layers[i - 1]])):
                            replicate = verticess[self.layers[i - 1]][k] * np.exp(PI2 / self.geo_atts[0] * 1j)
                            verticess[self.layers[i - 1]].append(replicate)

                blocked += conn[:, 0].tolist()

            for k, vertex in enumerate(poly[1:]):
                if k in blocked:
                    continue
                else:
                    verticess[self.layers[i]].append(vertex)

    def _polygen(self, polys):
        """
        Protected(!)
        Generator for iterating over polygons.
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

    def get_neighbors(self, index):
        """
        Get the neighbors of a polygon at index
        :param index: int = index of the polygon
        :return: np.array[p] = array containing the indices of the neighbors
        """
        # get equivalent poly in sector
        sector_replica = index // (self.sector_polys.shape[0] - 1)
        index %= (self.sector_polys.shape[0] - 1)
        jump = self.sector_polys.shape[0] - 1

        # quick and dirty solution
        neigbor_centers = util.generate_raw(self.sector_polys[index])
        indices = [self.find(e) for e in neigbor_centers]
        indices = [e for e in indices if not (e is False)]
        return [(i + sector_replica * jump) % (self.length - 1) if i != 0 else 0 for i in indices]

    def get_neighbors_fast(self, index):
        warnings.warn("NOT YET IMPLEMENTED")

        if index == 0:
            return np.array([1 + i * (self.sector_polys.shape[0] - 1) for i in range(self.geo_atts[0])])

        # placeholder for the neighbors
        neighbors = np.empty((self.geo_atts[0]), dtype=np.uint32)

        # map index to sector
        sector_index = index - 1
        sector_index = sector_index if sector_index < (self.sector_polys.shape[0] - 1) else sector_index % (
                    self.sector_polys.shape[0] - 1)
        c = 0

        # parent
        """
        Kann einen oder zwei parents geben
        2 nur, wenn es zwischen diesen liegt. Wenn es das tut, dann ist in edge_array die letzte ecke gesperrt
        """

        # siblings
        # only for q == 3, the polys have direct contact to their siblings. Otherwise q - 3 elements are between
        if self.geo_atts[1] == 3:
            neighbors[c] = self[sector_index - 1]
            neighbors[c + 1] = self[sector_index + 1]
            c += 2

        # children
        """
        offset der kinder bestimmen
        mit edge array bestimmen welche kinder noch involviert sind. 
        """

        return neighbors

    def check_integrity(self):
        """
        This function is numerically expensive!
        Checks the integrity of the grid. The number of neighbors as well as a search for duplicates is applied.
        Raises AttributeError if the grid seems to be invalid.
        :return: void
        """
        # check if one polygon is shifted in the range of another or if duplicates exist
        for i in range(len(self.sector_polys)):
            poly_center = self.sector_polys[i, 0]
            self.sector_polys[i, 0] = 0
            try:
                if self.find(poly_center):
                    raise AttributeError(f"Duplicate detected at index {i}")
            finally:
                self.sector_polys[i, 0] = poly_center

        # check if each layer has the correct size
        if self.layers is None:
            self.map_layers()

        for i, length in enumerate(self.sector_lengths):
            if np.count_nonzero(self.layers == i) != length:
                print(f"Layer {i} is not complete")

        # check if all edges have a partner
        for i in range(len(self.sector_polys)):
            neighbor_counter = len(self.get_neighbors(i))
            if neighbor_counter == self.geo_atts[0]:
                continue
            print(f"Integrity ensured till index {i} at layer {self.get_layer(i)}")
            return

    def transform(self, function):
        """
        Applies function to each polygon
        :param function: callable = function to apply on each polygon
        :return: void
        """
        if not isinstance(function, np.vectorize):
            function = np.vectorize(function)
        function(self.sector_polys)

    def rotate(self, angle):
        """
        Rotates the grid around angle
        :param angle: float = angle to rotate the polygon
        :return: void
        """
        self.transform(lambda x: trans.moeb_rotate_trafo(-angle, x))

    def translate(self, z):
        """
        Translates the grid to z
        :param z: complex = position of the new origin
        :return: void
        """
        self.transform(lambda x: trans.moeb_origin_trafo(x, z))


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    combis = [(7, 3), (3, 7), (5, 4), (4, 5), (6, 4), (7, 4), (7, 5), (7, 6), (7, 7)]
    ReflectTiling(3, 7, 2)

    for p, q in combis:
        print("\n\n")
        t1 = time.time()
        tiling = ReflectTiling(p, q, 4)
        print(time.time() - t1)
        # plot.plot(tiling, alpha=0.5)
        # tiling.map_layers()
        fig, ax = plt.subplots()

        colors = ["#0000FFAA", "#FF0000AA"]
        for i, pgon in enumerate(tiling):
            p_ = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), lw=1, edgecolor="#FFFFFF",
                                     fc=colors[tiling.get_layer(i) % 2])
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
    - non numba version
    - speed tests
"""
