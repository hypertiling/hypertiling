from typing import List
from hypertiling.kernel_abc import GraphExtended
import GRC_util as util
import numpy as np
import itertools


class GRC(GraphExtended):
    """
    GRC
    A GR based kernel working fully combinatorial
    """

    def __init__(self, p: int, q: int, n: int, sector=True, tiling=True, nbrs=True):
        super().__init__(p, q, n)

        self.sector = sector
        self.tiling = tiling
        self.nbrs = nbrs

        if sector:
            self.coords, self.nbrs_, self.lvls = util.construct_sector(p, q, n, self.tiling, self.nbrs)
            self.length = self.p * (self.lvls[-1] - 1) + 1
        else:
            self.coords, self.nbrs_, self.lvls = util.construct_full(p, q, n, self.tiling, self.nbrs)
            self.length = self.lvls[-1]

    def __getitem__(self, item):
        """
        Get neighbor of the polygon at index.

        Time-complexity (single polygon): O(p)

        Parameters
        ----------
        item : int
            Index of the polygon for whom the neighbors will be searched for.

        Returns
        -------
        np.array
            Indices of the neighbors.
        """
        if self.tiling:
            return self.get_coords(item)
        elif self.nbrs:
            return self.get_nbrs(item)
        else:
            TypeError("Neither nbrs nor coordinates were calculated!")

    def __len__(self):
        """
        Return the number of polygons in the tiling

        Time-complexity: O(1)

        Returns
        -------
        int
            Number of polygons in the tiling
        """
        return self.length

    def __iter__(self) -> np.array:
        """
        Iterates over the whole grid. As only one sector is stored in the memory, the others are generated when needed.
        Time-complexity (single polygon): O(p)
        :yield: np.array
            Array of shape [center, vertices].
        """
        if not self.tiling:
            raise AttributeError("Iterate is only available for tilings (i.e. tiling=True)")

        for poly in self.coords:
            yield poly

        if self.sector:
            dphi = 2 * np.pi / self.p
            phis = np.array([dphi * i for i in range(1, self.p)])
            for i, angle in enumerate(phis):
                for poly in self.coords[1:]:
                    yield poly * np.exp(angle * 1j)

    # Helper ###########################################################################################################

    def _map2fundamental(self, index):
        if index == 0:
            return 0, 0

        index -= 1
        k, index = divmod(index, self.lvls[-1] - 1)
        index += 1
        return k, index

    def _map2sector(self, index, k):
        jump = self.lvls[-1] - 1
        index = index + k * jump * np.clip(index, a_min=0, a_max=1)
        ks, index = np.divmod(index, self.length)
        return index + ks

    # Helper ###########################################################################################################

    def get_coords(self, index: int) -> np.complex128:
        """
        Get the coordinates for the center of the node at index.

        Time-complexity: O(1)

        Parameters
        ----------
        index : int
            Index of the node of consideration.

        Returns
        -------
        np.complex128
            Center of the node in complex coordinates.
        """
        if not self.tiling:
            AttributeError("Non tiling does not have coords (tiling=False)!")

        if self.sector:
            k, index = self._map2fundamental(index)

        coords = self.coords[index]
        if self.sector:
            return coords * np.exp(1j * k * np.pi * 2 / self.p)
        else:
            return coords

    def get_nbrs_list(self) -> List[List[int]]:
        """
        Create and return list of all neighbors

        Time-complexity: O(mp)

        Returns
        -------
        List[List[int]]
            List of all neighbors for all polygons.
        """
        if not self.nbrs:
            AttributeError("No neighbors as nbrs=False!")

        nbrs = self.nbrs_[:, 1:]
        if self.sector:
            nbrs = [nbrs_[np.where(nbrs_ != -1)] for nbrs_ in nbrs]
            sector_nbrs = [[self._map2sector(nbrs_, k).tolist() for nbrs_ in nbrs[1:]] for k in range(1, self.p)]
            nbrs = [nbrs_.tolist() for nbrs_ in nbrs]
            nbrs[0] = [(self.lvls[-1] - 1) * i + 1 for i in range(p)]
            return list(itertools.chain(*([nbrs] + sector_nbrs)))

        return [nbrs_[np.where(nbrs_ != -1)].tolist() for nbrs_ in nbrs]

    def get_nbrs(self, index: int) -> List[int]:
        """
        Create and return list of all neighbors of index

        Time-complexity: O(p)

        Returns
        -------
        List[int]
            List of all neighbors for polygon at index.
        """
        if not self.nbrs:
            AttributeError("No neighbors as nbrs=False!")

        if self.sector:
            k, index = self._map2fundamental(index)

        nbrs = self.nbrs_[index, 1:]
        nbrs = nbrs[np.where(nbrs != -1)]

        if self.sector:
            if index == 0:
                return np.array([(self.lvls[-1] - 1) * i + 1 for i in range(p)])
            else:
                return self._map2sector(nbrs, k)
        else:
            return nbrs

    def get_reflection_level(self, index) -> int:
        """
        Get the neighbors of a polygon at index

        Time-complexity: O(log(n + 1))

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        np.array
            Array containing the indices of the neighbors.
        """
        if self.sector:
            k, index = self._map2fundamental(index)

        level = np.searchsorted(self.lvls, index)
        return level + 1 if self.lvls[level] == index else level

    def check_integrity(self):
        raise NotImplementedError(
            "This function is no longer required (and supported) as the kernel is combinatorial now")


if __name__ == "__main__":
    import time
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    p, q, n = 7, 3, 5

    t1 = time.time()
    graph = GRC(p, q, n, sector=True)
    print(f"Took: {time.time() - t1}")
    nbrs = graph.get_nbrs_list()

    colors = ["#FF000060", "#00FF0060", "#0000FF60"]
    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)

    for i, poly in enumerate(graph):
        sector, _ = graph._map2fundamental(i)
        if sector != 0:
            continue

        facecolor = colors[graph.get_reflection_level(i) % len(colors)]
        patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in poly]), facecolor=facecolor,
                                    edgecolor="#FFFFFF")
        fig_ax[1].add_patch(patch)

        center = np.sum(poly) / p
        #for nbr in nbrs[i]:
        #    center2 = np.sum(graph.get_coords(nbr)) / p
        #    end = (center2 - center) / 2 + center
        #    fig_ax[1].plot((np.real(center), np.real(end)), (np.imag(center), np.imag(end)), color="#000000")
        fig_ax[1].text(np.real(center), np.imag(center), str(i + 1))

    plt.show()
