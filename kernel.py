import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from hypertiling.plot import plot_tiling, poly2patch

PI2 = 2 * np.pi


def to_real(poly):
    data = [(np.real(e), np.imag(e)) for e in poly]
    xs, ys = tuple(zip(*data))
    return xs, ys


def moeb_origin_trafo(z0, z):  # maps all points z such that z0 -> 0, leaves bounding |z|=1 circle invariant
    num = z - z0
    denom = 1 - z * np.conjugate(z0)
    return num / denom  # return coordinates of new point z'


def moeb_rotate_trafo(z, phi):  # rotates z by phi counter-clockwise about the origin
    return z * np.exp(complex(0, phi))


class ReflectionKernel:
    """
    Creates the hyperbolic tiling.
    """

    def __init__(self, p, q, n):
        """
        Initialize a hyperbolic tiling.
        TODO: Currently only for cell centered
        :param p: int = number of vertices per cells
        :param q: int = number of cells meeting at each vertex
        :param n: int =  number of layers to be constructed
        """

        # grid attributes
        if not ((p - 2) * (q - 2) > 4):
            raise AttributeError("Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")
        self.geo_atts = (p, q, n)

        # technical attributes
        self.sector_lengths = np.array([np.ceil(n / p) for n in self.get_ns()],
                                       dtype=np.int16)  # layer sizes of angular segment
        print(self.sector_lengths)
        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))

        # if center is added it should be p+1
        self.sector_polys = np.empty((np.sum(self.sector_lengths), p + 1), dtype=np.complex)
        self.generate()

    def get_ns(self):
        """
        Calculates the number of tildes the tiling will have.
        :return: np.array[int] = number of tildes per layer
        """
        lengths = np.empty((self.geo_atts[2],), dtype=np.uint16)
        lengths[0] = 0
        lengths[1] = (self.geo_atts[1] - 2) * self.geo_atts[0]
        fac = (self.geo_atts[1] - 2) * (self.geo_atts[0] - 2) - 2
        for i in range(2, self.geo_atts[2]):
            lengths[i] = fac * lengths[i - 1] - lengths[i - 2]

        lengths[0] = 1
        return lengths

    def generate(self):
        """
        Calculate the tilings polygons for an angular sector.
        :return: void
        """
        dphi = PI2 / self.geo_atts[0]
        phis = np.array([dphi * i for i in range(self.geo_atts[0])])

        # most inner polygon
        self.sector_polys[0, 0] = 0
        self.sector_polys[0, 1:] = self.r * np.exp(1j * phis)

        c = 1
        boundary = PI2 / self.geo_atts[0]
        for poly in self.sector_polys:
            for i, vertex in enumerate(poly[1:]):
                """
                Algorithm:
                 1. shift vertex into origin
                 2. rotate poly such that two vertices are on the x-axis
                 3. reflection on the x-axis (inversion of the imaginary part)
                 4. rotate poly back to original orientation (it is now reflected)
                 5. shift poly back to original position
                """
                z = moeb_origin_trafo(vertex, poly)
                phi = np.angle(z[1:][(i + 1) % self.geo_atts[0]])
                z = moeb_rotate_trafo(z, - phi)
                z = np.conjugate(z)
                z = moeb_rotate_trafo(z, phi)
                z = moeb_origin_trafo(- vertex, z)

                if 0 <= np.angle(z[0]) <= boundary and not (np.any(np.isclose(z[0], self.sector_polys[:c, 0]))):
                    self.sector_polys[c] = z
                    c += 1
                    if c == np.sum(self.sector_lengths):
                        print(c)
                        return 0
                    # check if edges of new polygon are blocked with boundary or not
                    # check if edges of new polygon are occupied by other polys
