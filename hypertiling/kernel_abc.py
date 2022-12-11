import abc
import numpy as np


class Graph(abc.ABC):

    def __init__(self, p: int, q: int, n: int, mangle: int):
        self.p = p
        self.q = q
        self.n = n

        # symmetry angles
        self.phi = 2 * np.pi / self.p  # angle of rotation that leaves the lattice invariant when cell centered
        self.qhi = 2 * np.pi / self.q  # angle of rotation that leaves the lattice invariant when vertex centered

        fac = np.pi / (p * q)
        self.r = np.sqrt(np.cos(fac * (p + q)) / np.cos(fac * (p - q)))

        self.mangle = mangle / 180 * np.pi

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

    """@abc.abstractmethod
    def get_nbrs(self, i):
        # TODO: implement default function
        pass"""

    """@abc.abstractmethod
    def get_nbrs_list(self):
        # TODO: implement default function
        pass"""
