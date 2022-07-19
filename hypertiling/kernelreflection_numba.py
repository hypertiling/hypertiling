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
def generate(geo_atts, r, sector_polys, sector_lengths, roll_f, sector_non_fillers, degtol):
    """
    Generates the tiling of the polygon
    :param geo_atts: Tuple[int, int, int] = [p, q, n]
    :param r: float = radius of the fundamental polygon
    :param sector_polys: np.array[complex][p + 1, x] = array containing the polygons [[center, vertices],...]
    :param sector_lengths: np.array[int] = length
    :param roll_f: callable = numba compiled callable for the correct ordering of the vertices in sector_polys
    :param sector_non_fillers: np.array[bool] = describes for each polygon if it is a filler polygon
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

            if np.angle(z[0]) > boundary:
                break

            elif 0 <= np.angle(z[0]):
                sector_polys[c, :] = z
                # has to be before if because of the "break" in the if
                c += 1

                if c == stop:
                    return 0

                # check if filler (shares edge with next polygon)
                if c > j + 2 and any_is_close(sector_polys[j + 1], z[its - 1]):
                    sector_non_fillers[c - 1] = 0
                    break

@njit()
def generate_raw(poly):
    """
    Generates the neigboring polygons for a single polygon poly
    :param poly: np.array[np.complex128][p + 1] = polygon to grow
    :return: np.array[np.complex128][p] = centers of the neigboring polygons
    """
    reflection_centers = np.empty((poly.shape[0] - 1, 1), dtype=np.complex128)
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
            self.length = np.sum(lengths)
            self.sector_lengths = np.ceil(lengths / p).astype(np.uint32)
        else:
            self.sector_lengths = np.array([1])

        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))
        self.degtol = degtol

        # some magic functions... I do not understand it 100% yet
        if self.geo_atts[0] == 3:
            self.roll_f = f1
        elif self.geo_atts[0] == 4:
            raise ArithmeticError("I have no clue. This one is evil :(")
        else:
            self.roll_f = f2

        # if center is added it should be p+1
        self.sector_polys = np.empty((np.sum(self.sector_lengths), p + 1), dtype=np.complex)
        self.sector_non_fillers = np.ones(np.sum(self.sector_lengths), dtype=np.bool)
        self.generate()

    def generate(self):
        """
        Calculate the tilings polygons for an angular sector.
        :return: void
        """
        generate(self.geo_atts, self.r, self.sector_polys, self.sector_lengths, self.roll_f, self.sector_non_fillers,
                 self.degtol)

    def __len__(self):
        """
        Return the number of created polygons
        :return: int = number of created polygons
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

    def find(self, x, y):
        """
        Find the polygons index (x, y) belongs to. x, y are the scalar of real and imaginary part.
        :param x: float = real part of the position
        :param y: float = imaginary part of the position
        :return: int = index of the corresponding polygon
        """
        v = x + y * 1j
        angle = np.angle(v)
        angle = angle if 0 <= angle else angle + PI2
        factor = angle // (PI2 / self.geo_atts[0])
        sector_proj = v * np.exp(-(factor * PI2 / self.geo_atts[0]) * 1j)

        disk_distance = np.vectorize(
            lambda z: 2 * np.arctanh(np.abs(z - sector_proj) / np.abs(1 - z * sector_proj.conjugate())))

        index = np.argmin(disk_distance(self.sector_polys[:, 0]))
        return int(index + (self.sector_polys.shape[0] - 1) * factor)

    def get_layer(self, index):
        index %= (self.sector_polys.shape[0] - 1)
        for l, length in enumerate(self.sector_lengths):
            index -= length
            if index < 0:
                return l

    def _polygen(self, polys):
        for poly in polys:
            yield poly

        dphi = PI2 / self.geo_atts[0]
        phis = np.array([dphi * i for i in range(1, self.geo_atts[0])])
        for i, angle in enumerate(phis):
            for poly in polys:
                yield poly * np.exp(angle * 1j)

    def get_polys_in_layer(self, layer, generator=True):
        start = np.sum(self.sector_lengths[:layer])
        stop = np.sum(self.sector_lengths[:layer + 1])
        if generator:
            return self._polygen(self.sector_polys[start:stop])
        else:
            return self.sector_polys[start:stop]

    def get_neighbors(self, index):
        # get equivalent poly in sector
        index %= (self.sector_polys.shape[0] - 1)
        neigbor_centers = generate_raw(self.sector_polys[index])
        xs, ys = list(zip(*[(np.real(e), np.imag(e)) for e in neigbor_centers]))
        plt.scatter(xs, ys)
        # quick and dirty solution
        return [self.find(np.real(e), np.imag(e)) for e in neigbor_centers]




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import time

    p, q, n = 7, 3, 5

    ReflectTiling(p, q, 2)  # for numba

    t1 = time.time()
    tiling = ReflectTiling(p, q, n)
    print(f"Creation of segment took: {time.time() - t1} s")

    # hps = []
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    colors = ["red", "blue"]
    print(tiling.sector_lengths)
    for i, pgon in enumerate(tiling):
        l = tiling.get_layer(i)
        p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), alpha=0.5, color=colors[l % 2])
        ax.add_patch(p)
        # ax.arrow(np.real(pgon[0]), np.imag(pgon[0]), np.real(pgon[1] - pgon[0]), np.imag(pgon[1] - pgon[0]), lw=1)
        ax.text(np.real(pgon[0]), np.imag(pgon[0]), i, fontsize=6, horizontalalignment='center',
                verticalalignment='center')

    # poly = tiling[199]
    # p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in poly[1:]]), alpha=0.5, color="black")
    # ax.add_patch(p)
    # print(tiling.find(np.real(poly[0]), np.imag(poly[0])))
    """l = tiling.get_layer(103)
    polys = tiling.get_polys_in_layer(l)
    for poly in polys:
        p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in poly[1:]]), alpha=0.5, color="black")
        ax.add_patch(p)"""

    neigbors = tiling.get_neighbors(199)
    for index in neigbors:
        p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in tiling[index][1:]]), alpha=0.5, color="black")
        ax.add_patch(p)

    plt.show()
