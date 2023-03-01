import abc
import numpy as np
from .ion import htprint
from .neighbors import find_radius_optimized, find_radius_optimized_single
from .util import lattice_spacing_weierstrass, fund_radius

# Magic number: transcendental number (Champernowne constant)
# used as an angular offset, rotates the entire construction slightly during construction
MAGICANGLE = np.radians(0.12345678910111213141516171819202122232425262728293031)


class Graph(abc.ABC):
    """ The abstract base class of a hyperbolic graph """

    def __init__(self, p: int, q: int, n: int, mangle: float = MAGICANGLE):

        if not ((p - 2) * (q - 2) > 4):
            raise AttributeError("Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

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
        self.mangle = mangle

        # a place to store adjaceny relations
        self._nbrs = None

    def __repr__(self):
        return f"Graph {self.p, self.q, self.n}"

    @abc.abstractmethod
    def __len__(self):
        pass


class Tiling(Graph):
    """ The abstract base class of a hyperbolic tiling """


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


    def get_nbrs_list(self, **kwargs):
        """
        calculates for each vertex the neighbours a returns a list.
        :return: List of list with neighbours
        """
        if self._nbrs is None:
            htprint("Status", "Mapping neighbours for entire lattice using 'optimized radius search' algorithm.")
            self._nbrs = find_radius_optimized(self, **kwargs)
        return self._nbrs


    def get_nbrs(self, i, **kwargs):
        """
        return the indices of the neighbours of vertex i.
        :param i: int the index you're intereted in.
        :return: a list of all neighbours
        """
        if self._nbrs is None:
            htprint("Status", "Performing radius search for one vertex. If neighbours of many points are required, we recommend to use 'get_nbrs_list'.")
            return find_radius_optimized_single(self, i, **kwargs)
        else:
            return self._nbrs[i]