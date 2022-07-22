import numpy as np
from numba import njit

PI2 = 2 * np.pi


@njit()
def moeb_origin_trafo(z, z0):  # maps all points z such that z0 -> 0, leaves bounding |z|=1 circle invariant
    """
    Shifts the origin of the points in z such that z0 -> 0. Leaves boundary |z| = 1 circle invariant
    :param z: np.array[complex] = array of points to shift
    :param z0: complex = new origin in the disc
    :result: np.array[complex] = array of shifted points
    """
    num = z - z0
    denom = 1 - z * np.conjugate(z0)
    return num / denom


@njit()
def moeb_rotate_trafo(z, phi):
    """
    Rotate the points described in z around the angle phi
    :param z: np.array[complex] = array of points to rotate
    :param phi: float = angle for the rotation
    :result: np.array[complex] = array of the rotated points
    """
    return z * np.exp(complex(0, phi))


@njit()
def any_is_close(zs, z, tol=1e-12):
    """
    Compares if the complex z is in the array zs, with tolerance tol
    :param zs: np.array[complex] = array with the floats to compare
    :param z: complex = the value to search for
    :param tol: float = tolerance of the comparison (absolut)
    :result: bool = True if float is in array else False
    """
    return np.any(np.abs(zs - z) <= tol)


@njit()
def generate(geo_atts, r, sector_polys, sector_lengths, roll_f, degtol):
    """
    Generates the tiling of the polygon
    :param geo_atts: Tuple[int, int, int] = [p, q, n]
    :param r: float = radius of the fundamental polygon
    :param sector_polys: np.array[complex][p + 1, x] = array containing the polygons [[center, vertices],...]
    :param sector_lengths: np.array[int] = length
    :param roll_f: callable = numba compiled callable for the correct ordering of the vertices in sector_polys
    :param degtol: float = tolerance at the boundary
    :return: void
    """
    dphi = PI2 / geo_atts[0]
    phis = np.array([dphi * i for i in range(geo_atts[0])])

    # most inner polygon
    sector_polys[0, 0] = 0
    sector_polys[0, 1:] = r * np.exp(1j * phis)

    c = 1
    its = max(int(np.ceil(geo_atts[0])) - 1, 3)
    its = its if geo_atts[0] - geo_atts[1] != 1 else its + 1
    stop = np.sum(sector_lengths)
    boundary = PI2 / geo_atts[0] + (degtol / 360 * PI2)
    ngeohalf = - (geo_atts[0] // 2 + 1)

    for j, poly in enumerate(sector_polys[:-1]):
        if j > 2 and any_is_close(sector_polys[c - 1, ngeohalf:], poly[2]):
            start = 2
        else:
            start = 1
        if any_is_close(sector_polys[j + 1], poly[2]):
            its_ = its - 1
        else:
            its_ = its

        for k, vertex in enumerate(poly[start:its_]):
            i = k + start
            """
            Algorithm:
             1. shift vertex into origin
             2. rotate poly such that two vertices are on the x-axis
             3. reflection on the x-axis (inversion of the imaginary part)
             4. rotate poly back to original orientation (it is now reflected)
             5. shift poly back to original position
            """
            z = moeb_origin_trafo(poly, vertex)
            phi = np.angle(z[1:][i % geo_atts[0]])
            z = moeb_rotate_trafo(z, - phi)
            z = np.conjugate(z)
            z = moeb_rotate_trafo(z, phi)
            z = moeb_origin_trafo(z, - vertex)
            z[1:] = roll_f(z, i)

            angle = np.angle(z[0])
            if np.angle(z[0]) > boundary:
                break

            if 0 < angle:
                sector_polys[c, :] = z
                # has to be before if because of the "break" in the if
                c += 1

                if c == stop:
                    return 0

                # check if filler (shares edge with next polygon)
                if c > j + 2 and any_is_close(sector_polys[j + 1], z[its - 1]):
                    break


@njit()
def generate_raw(poly):
    """
    Generates the neigboring polygons for a single polygon poly
    :param poly: np.array[np.complex128][p + 1] = polygon to grow
    :return: np.array[np.complex128][p] = centers of the neigboring polygons
    """
    reflection_centers = np.empty((poly.shape[0] - 1,), dtype=np.complex128)
    for k, vertex in enumerate(poly[1:]):
        z = moeb_origin_trafo(poly, vertex)
        phi = np.angle(z[1:][(k + 1) % (poly.shape[0] - 1)])
        z = moeb_rotate_trafo(z[0], - phi)
        z = np.conjugate(z)
        z = moeb_rotate_trafo(z, phi)
        z = moeb_origin_trafo(z, - vertex)
        reflection_centers[k] = z
    return reflection_centers


@njit()
def get_ns(geo_atts):
    """
    Calculates the number of tildes the tiling will have.
    :param geo_atts: Tuple[int, int, int] = (p, q, n)
    :return: np.array[int] = number of tildes per layer
    """
    lengths = np.empty((geo_atts[2],), dtype=np.uint32)
    lengths[0] = 0
    lengths[1] = (geo_atts[1] - 2) * geo_atts[0]
    fac = (geo_atts[1] - 2) * (geo_atts[0] - 2) - 2
    for i in range(2, geo_atts[2]):
        lengths[i] = fac * lengths[i - 1] - lengths[i - 2]

    lengths[0] = 1
    return lengths


@njit()
def f1(z, i):
    """
    Ensures the order of the vertices for p > 4.
    :param z: np.array[complex][8] = [center, vertices]
    :param i: int = i-th child of the parent polygon
    :return: void
    """
    return np.roll(np.flip(z[1:]), i)


@njit()
def f2(z, i):
    """
    Ensures the order of the vertices for p > 4.
    :param z: np.array[complex][8] = [center, vertices]
    :param i: int = i-th child of the parent polygon
    :return: void
    """
    return np.roll(np.flip(z[1:]), i - 1)


@njit()
def f3(z, i):
    return np.roll(np.flip(z[1:]), i)


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
            lengths = get_ns(self.geo_atts)
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
        if self.geo_atts[0] == 3:
            print("f1")
            self.roll_f = f1
        elif abs(diff) == 1:
            print("f2")
            self.roll_f = f3
        else:
            print("f3")
            self.roll_f = f2

        # if center is added it should be p+1
        self.sector_polys = np.empty((np.sum(self.sector_lengths), p + 1), dtype=np.complex)
        self.generate()

    def generate(self):
        """
        Calculate the tilings polygons for an angular sector.
        :return: void
        """
        generate(self.geo_atts, self.r, self.sector_polys, self.sector_lengths, self.roll_f, self.degtol)

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

        """dphi = PI2 / self.geo_atts[0]
        phis = np.array([dphi * i for i in range(1, self.geo_atts[0])])
        for i, angle in enumerate(phis):
            for poly in self.sector_polys[1:]:
                yield poly * np.exp(angle * 1j)"""

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
        neigbor_centers = generate_raw(self.sector_polys[index])
        indices = [self.find(np.real(e), np.imag(e)) for e in neigbor_centers if not (e is False)]
        return [(i + sector_replica * jump) % (self.length - 1) if i != 0 else 0 for i in indices]

    def check_integrity(self):
        # check if one polygon is shifted in the range of another or if duplicates exist
        for i in range(len(self.sector_polys)):
            poly_center = self.sector_polys[i, 0]
            self.sector_polys[i, 0] = 0
            try:
                if self.find(poly_center):
                    raise AttributeError(f"Duplicate detected at index {i}")
            finally:
                self.sector_polys[i, 0] = poly_center

        # TODO: check for rotational duplicates
        # check if all edges have a partner
        pass

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
        self.transform(lambda x: moeb_rotate_trafo(x, - angle))

    def translate(self, z):
        """
        Translates the grid to z
        :param z: complex = position of the new origin
        :return: void
        """
        self.transform(lambda x: moeb_origin_trafo(x, z))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.animation as animation
    import time
    import random

    p, q, n = 4, 5, 4
    combis = [(7, 3), (3, 7), (5, 4), (4, 5), (6, 4), (7, 4), (7, 5), (7, 6), (7, 7)]

    for p, q in combis:
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{p} {q}")
        tiling = ReflectTiling(p, q, 4)  # for numba
        try:
            tiling.check_integrity()
            check = True
        except:
            check = False
        ax.set_title(f"{p} {q} {check}")
        for pgon in tiling:
            # l = tiling.get_layer(i)
            # p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5, color=colors[l % 2])
            p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5)
            ax.add_patch(p)
        plt.show()
    exit("stop")

    t1 = time.time()
    tiling = ReflectTiling(p, q, n)
    print(f"Creation of segment took: {time.time() - t1} s")
    tiling.check_integrity()

    # hps = []
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    colors = ["red", "blue"]

    for i, pgon in enumerate(tiling):
        # l = tiling.get_layer(i)
        # p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5, color=colors[l % 2])
        p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5)
        ax.add_patch(p)
        ax.text(np.real(pgon[0]), np.imag(pgon[0]), i, fontsize=6, horizontalalignment='center',
                verticalalignment='center')
    j = random.randint(0, i)
    poly = tiling[j]
    p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in poly[1:]]), alpha=0.5)
    ax.add_patch(p)
    print(f"{j}: {tiling.find(poly[0])}")

    """frames = []
    for j in range(len(tiling)):
        temp = []
        for i, pgon in zip(range(j), tiling):
            # l = tiling.get_layer(i)
            # p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5, color=colors[l % 2])
            p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5)
            temp.append(ax.add_patch(p))
            d = pgon[1] - pgon[0]
            temp.append(ax.arrow(np.real(pgon[0]), np.imag(pgon[0]), np.real(d), np.imag(d)))
            # ax.text(np.real(pgon[0]), np.imag(pgon[0]), i, fontsize=6, horizontalalignment='center',
            #        verticalalignment='center')
        frames.append(temp)"""

    """neigbors = tiling.get_neighbors(199)  # 199
    for index in neigbors:
        p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in tiling[index][1:]]), alpha=0.5,
                                color="black")
        ax.add_patch(p)
    ani = animation.ArtistAnimation(fig, frames, interval=500, blit=False)"""
    plt.show()

"""
Arbeitsplan:
    - generate soll für alle funktionieren
    - neighbour function fertig schreiben
"""
