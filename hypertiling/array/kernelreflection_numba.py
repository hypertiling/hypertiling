import numpy as np
import reflection_numba_util as util
from reflection_numba_util import PI2

# FIXME: remove me later
import warnings


class ReflectTiling:
    """
    Creates the hyperbolic tiling.
    """

    def __init__(self, p, q, n, degtol=1):
        """
        Initialize a hyperbolic tiling. CELL CENTERED ONLY!
        :param p: int = number of vertices per cells
        :param q: int = number of cells meeting at each vertex
        :param n: int =  number of layers to be constructed
        :param degtol: int = tolerance at boundary in degrees (1 for (3,7), (7, 3) but 15 for (5, 4))
        """

        # grid attributes
        if not ((p - 2) * (q - 2) > 4):
            raise AttributeError("Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")
        self.geo_atts = (p, q, n)

        # technical attributes
        if n > 1:
            lengths = util.get_ns(self.geo_atts)
            self.sector_lengths = np.ceil(lengths / p).astype(np.uint32)
            self.sector_commulated_length = np.empty((self.sector_lengths.shape[0],), dtype=np.uint32)
            value = 0
            for i, length in enumerate(self.sector_lengths):
                self.sector_commulated_length[i] = value
                value += length

        else:
            self.sector_lengths = np.array([1])
            self.sector_commulated_length = np.array([0])
        self.length = np.sum(lengths)

        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))
        self.degtol = degtol

        # some magic functions... I do not understand it 100% yet
        diff = self.geo_atts[0] - self.geo_atts[1]
        print(f"{p}, {q}: ", end="")
        if self.geo_atts[0] == 3 or abs(diff) == 1:
            print("f1")
            self.roll_f = util.f1
        else:
            print("f3")
            self.roll_f = util.f2

        # if center is added it should be p+1
        self.sector_polys = np.empty((np.sum(self.sector_lengths), p + 1), dtype=np.complex)
        self.generate()

    def generate(self):
        """
        Calculate the tilings polygons for an angular sector.
        :return: void
        """
        util.generate(self.geo_atts, self.r, self.sector_polys, self.sector_lengths, self.roll_f, self.degtol)

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

        phi = PI2 / self.geo_atts[0] * (index // (self.sector_polys.shape[0] - 1))
        poly = self.sector_polys[index % (self.sector_polys.shape[0] - 1)]
        return poly * np.exp(phi * 1j)

    def find(self, v):
        """
        Find the polygons index z belongs to.
        :param z: complex = position to search polygon for
        :return: int = index of the corresponding polygon
        """
        angle = np.angle(v)
        factor = (angle - self.degtol / 360 * PI2) // (PI2 / self.geo_atts[0])
        factor = factor if factor >= 0 else factor + self.geo_atts[0]
        sector_proj = v * np.exp(-(factor * PI2 / self.geo_atts[0]) * 1j)

        f_dist = lambda z, z_hat: 2 * np.arctanh(np.abs(z - z_hat) / np.abs(1 - z * z_hat.conjugate()))
        disk_distance = np.vectorize(lambda z: f_dist(z, sector_proj))

        dists = disk_distance(self.sector_polys[:, 0])
        index = np.argmin(dists)

        if dists[index] >= f_dist(self.sector_polys[0, 0], self.sector_polys[1, 0]) / 2:
            return False
        return int(index + (self.sector_polys.shape[0] - 1) * factor if index != 0 else 0)

    def get_layer(self, index):
        """
        Experimental! Get layer of the polygon at index.
        The layer is determined by the number of polygons per layer. This causes problem in e.g. (3, 7).
        :param index: int = index of the polygon
        :return: int = layer the polygon belongs to
        """
        warnings.warn("Experimental method! This might not work for certain grids")
        index %= (self.sector_polys.shape[0] - 1)
        for l, length in enumerate(self.sector_lengths):
            index -= length
            if index < 0:
                return l

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

    def get_polys_in_layer(self, layer, generator=True):
        """
        Experimental! Get all polygons in a certain layer.
        The layer is determined by the number of polygons per layer. This causes problem in e.g. (3, 7).
        :param layer: int = index of the layer
        :param generator: bool = determines if the rotational duplicates should be considered too (returns generator)
        :return: Union[iterable, np.array] = generator or segment of all the polygons in the layer
        """
        warnings.warn("Experimental method! This might not work for certain grids")
        if generator:
            return self._polygen(
                self.sector_polys[self.sector_commulated_length[layer]:self.sector_commulated_length[layer + 1]])
        else:
            return self.sector_polys[self.sector_commulated_length[layer]:self.sector_commulated_length[layer + 1]]

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
        pass

    def check_integrity(self):
        """
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

        # check if all edges have a partner
        # FIXME: Check radial und melde "erstes loch in radius... found"
        warnings.warn("check_integrity is not yet ready. Will cause problems in e.g. 3,7-grid with boundary")
        for i in range(len(self.sector_polys)):
            neighbor_counter = len(self.get_neighbors(i))
            if (i >= self.sector_commulated_length[-1] and neighbor_counter >= 3) or neighbor_counter == self.geo_atts[
                0]:
                continue
            raise AttributeError(f"A hole in the tiling was detected by {i}")

    def transform(self, function):
        """
        Applies function to each polygon
        :param function: callable = function to apply on each polygon
        :return: void
        """
        if not isinstance(function, np.vectorize):
            function = np.vectorize(function)
        self.sector_polys = function(self.sector_polys)

    def rotate(self, angle):
        """
        Rotates the grid around angle
        :param angle: float = angle to rotate the polygon
        :return: void
        """
        self.transform(lambda x: util.moeb_rotate_trafo(x, - angle))

    def translate(self, z):
        """
        Translates the grid to z
        :param z: complex = position of the new origin
        :return: void
        """
        self.transform(lambda x: util.moeb_origin_trafo(x, z))


if __name__ == "__main__":
    import plot
    import matplotlib.pyplot as plt

    combis = [(7, 3), (3, 7), (5, 4), (4, 5), (6, 4), (7, 4), (7, 5), (7, 6), (7, 7)]

    for p, q in combis:
        tiling = ReflectTiling(p, q, 3)
        fig, ax = plot.plot(tiling, numerate=True, alpha=0.5)
        try:
            tiling.check_integrity()
            check = True
        except Exception as error:
            check = str(error)
        ax.set_title(f"{p} {q} {check}")

        plt.show()
"""
Arbeitsplan:
    - check integrity fertig machen
    - generate soll für alle funktionieren
    - find function fertig machen
"""
