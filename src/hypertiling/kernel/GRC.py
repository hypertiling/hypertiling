from typing import List, Tuple
from hypertiling.kernel_abc import Graph
import hypertiling.kernel.GRC_util as util
import hypertiling.ion as ion
import numpy as np
import itertools


class GRC(Graph):
    """
    Generative Reflection Combinatorial
    GR based kernel for generating (p, q, n) tilings
    """

    def __init__(self, p: int, q: int, n: int, sector: bool = True, tiling: bool = False, nbrs: bool = False):
        """
        Initialize a tesselation with Schwarian triangles.

        Parameters
        ----------
        p : int
            Number of edges per cell
        q : int
            Number of cells meeting at a vertex
        n : int
            Number of layers to be constructed
        sector : bool
            If True, a single sector is constructed and stored. All other sectors are provided on-demand by generation.
            (Not yet implemented)!
        tiling : bool
            If True, coordinates are calculated for the triangles
        nbrs : bool
            If True, neighbor relations are traced during construction
        """
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
        If tiling=True, returns the coordinates of the triangle at index {item}
        If tiling=False, nbrs=True, returns the neighbor array for the triangle at index {item}
        Requires at least one of tiling or nbrs to be True!
        Parameters
        ----------
        item : int
            Index of the polygon.

        Returns
        -------
        np.array
            Either array of shape 3 yielding coordinates
            OR
            array of shape 3 yielding neighbor relations
        """
        if self.tiling:
            return self.get_vertices(item)
        elif self.nbrs:
            return self.get_nbrs(item)
        else:
            raise TypeError("Neither nbrs nor coordinates were calculated!")

    def __len__(self):
        """
        Return the number of polygons in the tiling.

        Time-complexity: O(1)

        Returns
        -------
        int
            Number of cells in the tiling.
        """
        return self.length

    def __iter__(self) -> np.array:
        """
        Iterates over the tiling and yields the triangle coordinates.
        This is not available if tiling=False!
        :yield: np.array
            Array of shape [center, vertices]
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

    def _map2fundamental(self, index: int) -> Tuple[int, int]:
        """
        Map index to index of corresponding cell in the fundamental sector

        Parameters
        ----------
        index : int
            Index of a cell in the tesselation

        Returns
        -------
        int
            Index of the corresponding cell in the fundamental sector
        int
            Number indicating the sector the original cell is in
        """
        if index == 0:
            return 0, 0

        index -= 1
        k, index = divmod(index, self.lvls[-1] - 1)
        index += 1
        return k, index

    def _map2sector(self, index: int, k: int) -> int:
        """
        Calculate index for a corresponding cell in sector {k} when given a cell with index {index} in the fundamental
        sector

        Parameters
        ----------
        int
            Index of the corresponding cell in the fundamental sector
        int
            Number indicating the sector the original cell is in

        Returns
        -------
        int
            Index of the corresponding cell in the k-th sector

        """
        jump = self.lvls[-1] - 1
        index = index + k * jump * np.clip(index, a_min=0, a_max=1)
        ks, index = np.divmod(index, self.length)
        return index + ks

    # Helper ###########################################################################################################

    def get_vertices(self, index: int) -> np.complex128:
        """
        Returns the p vertices of the polygon at index.
        Requires tiling=True!

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        np.array
            Array of shape (p,) containing vertices of the polygon.
        """
        if not self.tiling:
            raise AttributeError("Non tiling does not have coords (tiling=False)!")

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
            raise AttributeError("No neighbors as nbrs=False!")

        nbrs = self.nbrs_[:, 1:]
        if self.sector:
            nbrs = [nbrs_[np.where(nbrs_ != -1)] for nbrs_ in nbrs]
            sector_nbrs = [[self._map2sector(nbrs_, k).tolist() for nbrs_ in nbrs[1:]] for k in range(1, self.p)]
            nbrs = [nbrs_.tolist() for nbrs_ in nbrs]
            nbrs[0] = [(self.lvls[-1] - 1) * i + 1 for i in range(self.p)]
            return list(itertools.chain(*([nbrs] + sector_nbrs)))

        return [nbrs_[np.where(nbrs_ != -1)].tolist() for nbrs_ in nbrs]

    def get_nbrs(self, index: int) -> np.array:
        """
        Create and return list of all neighbors of index

        Time-complexity: O(p)

        Returns
        -------
        np.array[int]
            Array of all neighbors for polygon at index.
        """
        if self.nbrs is False:
            raise AttributeError("No neighbors as nbrs=False!")

        if self.sector:
            k, index = self._map2fundamental(index)

        nbrs = self.nbrs_[index, 1:]
        nbrs = nbrs[:self.nbrs_[index, 0]]
        # nbrs = nbrs[np.where(nbrs != -1)]

        if self.sector:
            if index == 0:
                return np.array([(self.lvls[-1] - 1) * i + 1 for i in range(self.p)])
            else:
                return self._map2sector(nbrs, k)
        else:
            return nbrs

    def get_reflection_level(self, index: int) -> int:
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
        """
        Controls the integrity of the tesselation by
        1. Controlling the number of neighbors for each cell
        2. Bidirectional search for dual lattice
        Time-complexity: O(TODO)
        :return: void
        """

        if not self.nbrs:
            raise AttributeError("Non-Graph (nbrs=False) has no check_integrity implementation")

        n = self.n - ((self.q + 1) // 2)
        ion.htprint("Status", f"Integrity can only be ensured for the first n - q // 2 layer and thus layer {n}")

        # check for nbrs
        ln = self.lvls[self.n - 2] - 1
        for i in range(self.lvls[self.n - 2]):
            progbar = ">" * (l := int(64 * i / ln)) + " " * (64 - l)
            ion.htprint("Status", f"\r|{progbar}| Controlling nbrs for {i} / {ln}", end="")
            n_ = self.get_reflection_level(i)

            nbrs = self.get_nbrs(i)
            if len(nbrs) != len(set(nbrs)):
                raise AttributeError(f"Tesselation is corrupted! At least one nbr is registered at least twice {nbrs}!")
            elif len(nbrs) < self.p:
                raise AttributeError(
                    f"Tesselation is corrupted! Cell {i} in layer {n_} has {len(nbrs)}/{self.p} unique nbrs")

        ion.htprint("Status", f"\r|{progbar}| Controlling nbrs for {ln} / {ln}", end="\n")
        # print("")

        # dual lattice
        ln = self.lvls[n - 1]  # - 1

        for i in range(self.lvls[n - 1]):
            progbar = ">" * (l := int(64 * i / ln)) + " " * (64 - l)
            ion.htprint("Status", f"\r|{progbar}| Bidirectional search for cell {i} / {ln}", end="")

            # print(f"Cell {i} has nbrs {self.get_nbrs(i)}")
            for nbr in self.get_nbrs(i):
                # print(f"\nSearch {i} - {nbr}")
                # bidirectional search with depth (q - 1) // 2 + 1/2 if q is even
                cells = [i]
                cell_parents = {i: nbr}
                nbr_cells = [nbr]
                nbr_parents = {nbr: i}
                # print(nbr_cells, cells)
                for s in range(end := self.q // 2):

                    new_cells = []
                    # print("\t", cells, end=">")
                    for cell in cells:
                        new_cells += (childs := [nbr for nbr in self.get_nbrs(cell) if nbr != cell_parents[cell]])
                        cell_parents |= {child: cell for child in childs}
                    cells = new_cells
                    # print("\t", cells)

                    # print("\t", nbr_cells, end=" >")
                    if not (self.q & 1 == 0 and s + 1 == end):
                        new_nbrs = []
                        for cell in nbr_cells:
                            new_nbrs += (childs := [nbr for nbr in self.get_nbrs(cell) if nbr != nbr_parents[cell]])
                            nbr_parents |= {child: cell for child in childs}
                        nbr_cells = new_nbrs

                    # print("\t", nbr_cells)
                    # print(s, end, nbr_cells, cells)
                    if sum([1 if (cell in nbr_cells) else 0 for cell in cells]) == 2:
                        break
                else:
                    raise ArithmeticError(f"At least one connection between {i} - {nbr} could not be established!")
            # exit()
        progbar = ">" * 64
        ion.htprint("Status", f"\r|{progbar}| Bidirectional search for cell {ln} / {ln}", end="\n")
        ion.htprint("Status", f"Integrity ensured!")


if __name__ == "__main__":
    import time
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    ion.set_verbosity_level("Status")

    p, q, n = 3, 8, 6  # 11

    t1 = time.time()
    graph = GRC(p, q, n, sector=False, nbrs=True, tiling=True)
    print(f"Took: {time.time() - t1}")

    try:
        graph.check_integrity()
    except:
        pass

    # exit()
    nbrs = graph.get_nbrs_list()

    colors = ["#FF000060", "#00FF0060", "#0000FF60"]
    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)

    for i, poly in enumerate(graph):
        sector, _ = graph._map2fundamental(i)
        # if sector != 0:
        #    continue

        #9612 - 22076
        facecolor = colors[graph.get_reflection_level(i) % len(colors)]
        patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in poly[1:]]), facecolor=facecolor,
                                    edgecolor="#FFFFFF")
        fig_ax[1].add_patch(patch)

        center = poly[0]
        # for nbr in nbrs[i]:
        #   center2 = graph.get_vertices(nbr)[0]
        #   end = (center2 - center) / 2 + center
        #   fig_ax[1].plot((np.real(center), np.real(end)), (np.imag(center), np.imag(end)), color="#000000")
        # if i in [9612, 22076]:
        fig_ax[1].text(np.real(center), np.imag(center), str(i))

    plt.show()
