import numpy as np
from numba import njit
import matplotlib.pyplot as plt


PI2 = 2 * np.pi


@njit()
def moeb_origin_trafo(z0, z):  # maps all points z such that z0 -> 0, leaves bounding |z|=1 circle invariant
    num = z - z0
    denom = 1 - z * np.conjugate(z0)
    return num / denom  # return coordinates of new point z'


@njit()
def moeb_rotate_trafo(z, phi):  # rotates z by phi counter-clockwise about the origin
    return z * np.exp(complex(0, phi))


@njit()
def any_is_close(x, y, tol=1e-8):
    return np.any(np.abs(y - x) <= tol)


@njit()
def generate(geo_atts, r, sector_polys, sector_lengths, roll_f, sector_non_fillers):
    dphi = PI2 / geo_atts[0]
    phis = np.array([dphi * i for i in range(geo_atts[0])])

    # most inner polygon
    sector_polys[0, 0] = 0
    sector_polys[0, 1:] = r * np.exp(1j * phis)

    c = 1
    its = max(int(np.ceil(geo_atts[0])) - 1, 3)
    stop = np.sum(sector_lengths)
    boundary = (PI2 + 1 / PI2) / geo_atts[0]

    for j, poly in enumerate(sector_polys[:-1]):
        if j > 2 and any_is_close(poly[2], sector_polys[c - 1, -4:]):
            start = 2
        else:
            start = 1
        if any_is_close(poly[2], sector_polys[j + 1]):
            its_ = its - 1
        else:
            its_ = its
        for k, vertex in enumerate(poly[start:its_]):
            i = k + start
            # temp = []
            """
            Algorithm:
             1. shift vertex into origin
             2. rotate poly such that two vertices are on the x-axis
             3. reflection on the x-axis (inversion of the imaginary part)
             4. rotate poly back to original orientation (it is now reflected)
             5. shift poly back to original position
            """
            z = moeb_origin_trafo(vertex, poly)
            phi = np.angle(z[1:][i % geo_atts[0]])
            z = moeb_rotate_trafo(z, - phi)
            z = np.conjugate(z)
            z = moeb_rotate_trafo(z, phi)
            z = moeb_origin_trafo(- vertex, z)
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
                if c > j + 2 and any_is_close(z[its - 1], sector_polys[j + 1]):
                    sector_non_fillers[c - 1] = 0
                    break


@njit()
def get_ns(geo_atts):
    """
    Calculates the number of tildes the tiling will have.
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
    return np.roll(np.flip(z[1:]), i if i > 1 else 1)


@njit()
def f2(z, i):
    return np.roll(np.flip(z[1:]), i - 1 if i > 1 else 0)


class ReflectTiling:
    """
    Creates the hyperbolic tiling.
    """

    def __init__(self, p, q, n):
        """
        Initialize a hyperbolic tiling. CELL CENTERED ONLY (FOR NOW)!
        :param p: int = number of vertices per cells
        :param q: int = number of cells meeting at each vertex
        :param n: int =  number of layers to be constructed
        """

        # grid attributes
        if not ((p - 2) * (q - 2) > 4):
            raise AttributeError("Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")
        self.geo_atts = (p, q, n)

        # technical attributes
        if n > 1:
            self.sector_lengths = np.ceil(get_ns(self.geo_atts) / p).astype(np.uint32)
        else:
            self.sector_lengths = np.array([1])

        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))

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
        generate(self.geo_atts, self.r, self.sector_polys, self.sector_lengths, self.roll_f, self.sector_non_fillers)


if __name__ == "__main__":
    import hypertiling.hyperpolygon as hy
    import hypertiling.plot as hp
    import matplotlib.pyplot as plt
    import time
    import random

    p, q, n = 7, 3, 12
    # for numba to compile the functions
    ReflectTiling(p, q, 2)

    t1 = time.time()
    tiling = ReflectTiling(p, q, n)
    print(f"Creation of segment took: {time.time() - t1} s")

    hps = []
    for pgon in tiling.sector_polys:
        newpoly = hy.HyperPolygon(tiling.geo_atts[0])
        newpoly.verticesP = np.array(pgon[1:].tolist() + [pgon[0]])
        hps.append(newpoly)

    print(len(hps))
    hp.plot_tiling(hps, [random.random() for i in range(len(hps))])

    for pgon in tiling.sector_polys:
        to_vertex = pgon[1] - pgon[0]
        plt.arrow(np.real(pgon[0]), np.imag(pgon[0]), np.real(to_vertex), np.imag(to_vertex))

    plt.show()
