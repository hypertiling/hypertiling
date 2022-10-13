import abc
import numpy as np


class AbstractKernelBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, p: int, q: int, n: int):
        pass

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
