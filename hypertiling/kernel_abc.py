import abc
import numpy as np

from .neighbors import find_radius_optimized
from .util import lattice_spacing_weierstrass, fund_radius

# Magic number: transcendental number (Champernowne constant)
# used as an angular offset, rotates the entire construction by a bit during construction
MAGICANGLE = np.radians(5.1234567891011121314151617181920212223242526272829303132333)


class Graph(abc.ABC):

    def __init__(self, p: int, q: int, n: int, mangle: float = MAGICANGLE):

        # fundamental lattice parameters
        self.p = p
        self.q = q
        self.n = n

        # symmetry angles
        self.phi = 2 * np.pi / self.p  # angle of rotation that leaves the lattice invariant when cell centered
        self.qhi = 2 * np.pi / self.q  # angle of rotation that leaves the lattice invariant when vertex centered

        # radius of the fundamental polygon in the Poincare disk
        self.r = fund_radius(self.p, self.q)

        # hyperbolic/geodesic lattice spacing, i.e. the edge length of any cell
        self.h = lattice_spacing_weierstrass(self.p, self.q)

        # geodesic radius (i.e. distance between center and any vertex) of cells in a regular p,q tiling
        self.hr = lattice_spacing_weierstrass(self.q, self.p)

        # magic angle required for technical reasons
        self.mangle = mangle / 180 * np.pi

        # a place to store adjaceny relations
        self._nbrs = None

    def __repr__(self):
        return f"Graph {self.p, self.q, self.n}"

    @abc.abstractmethod
    def get_nbrs(self, i):
        pass

    @abc.abstractmethod
    def get_nbrs_list(self):
        pass

    """@abc.abstractmethod
    def check_integrity(self):
        pass"""


class Tiling(Graph):

    def __repr__(self):
        return f"Tiling {self.p, self.q, self.n}"

    @abc.abstractmethod
    def get_layer(self, index: int) -> int:
        """
        Returns the layer to the center of the polygon at index.
        :param index: int = index of the polygon
        :return: int = layer of the polygon
        """
        pass

    @abc.abstractmethod
    def get_sector(self, index: int) -> int:
        """
        Returns the sector, the polygon at index refers to.
        :param index: int = index of the polygon
        :return: int = number of the sector
        """
        pass

    @abc.abstractmethod
    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index.
        :param index: int = index of the polygon
        :return: np.complex128 = center of the polygon
        """
        pass

    @abc.abstractmethod
    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index.
        :param index: int = index of the polygon
        :return: np.array[np.complex128][p] = vertices of the polygon
        """
        pass

    @abc.abstractmethod
    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.
        :param index: int = index of the polygon
        :return: float = center of the polygon
        """
        pass


    def get_nbrs(self, i):
        if self._nbrs is None:
            print("start mapping neighbors")
            self._nbrs = find_radius_optimized(self)
        return self._nbrs[i]
        

    def get_nbrs_list(self):
        self._nbrs = find_radius_optimized(self)
        return self._nbrs