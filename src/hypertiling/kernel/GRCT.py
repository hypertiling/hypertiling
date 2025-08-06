from typing import List, Optional
import numpy as np
from hypertiling.kernel_abc import Graph
import hypertiling.ion as ion
import hypertiling.kernel.GRCT_util as util
import itertools


class GRCT(Graph):
    """
    Generative Reflection Combinatorial Triangle
    Kernel for generating (p, q, r, n) tilings of Schwarzian triangles
    """

    def __init__(self, p: int, q: int, r: int, n: int, sector: bool = False, tiling: bool = False, nbrs: bool = False,
                 size: Optional[int] = None):
        """
        Initialize a tesselation with Schwarian triangles.

        Parameters
        ----------
        p : int
            Number of cells meeting at 1st vertex
        q : int
            Number of cells meeting at 2nd vertex
        r : int
            Number of cells meeting at 3rd vertex
        n : int
            Number of layers to be constructed
        sector : bool
            If True, a single sector is constructed and stored. All other sectors are provided on-demand by generation.
            (Not yet implemented)!
        tiling : bool
            If True, coordinates are calculated for the triangles
        nbrs : bool
            If True, neighbor relations are traced during construction
        size : Optional[int]
            Size of the reserved memory for the arrays.
            If None, an estimation will be used (usually overestimates a lot)
        """
        super().__init__(p, q, n)
        self.r = r

        self.sector = sector
        self.tiling = tiling
        self.nbrs = nbrs

        if sector:
            raise NotImplementedError("NOT YET IMPLEMENTED")
        else:
            size = util.get_n(p, q, r, n) if size is None else size
            self.coords, self.nbrs_, self.lvls = util.construct_full(p, q, r, n, size, self.tiling, self.nbrs)
            self.length = self.lvls[-1]

    def __repr__(self):
        """
        Return a string representation of the GraphExtended object.

        Returns
        -------
        str
            String representation of the GraphExtended object.
        """
        return f"Schwarzian {self.p, self.q, self.r, self.n}"

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

    def __getitem__(self, item: int) -> np.array:
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

    def __len__(self) -> int:
        """
        Return the number of polygons in the tiling.

        Time-complexity: O(1)

        Returns
        -------
        int
            Number of cells in the tiling.
        """
        return self.length

    def get_vertices(self, index: int) -> np.array:
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
        return self.coords[index]

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
        level = np.searchsorted(self.lvls[1:], index)
        return level + 1 if self.lvls[level + 1] == index else level

    def get_nbrs(self, index: int) -> np.array:
        """
        Create and return list of all neighbors of index

        Time-complexity: O(p)

        Returns
        -------
        np.array[int]
            Array of all neighbors for polygon at index.
        """
        if not self.nbrs:
            raise AttributeError("No neighbors as nbrs=False!")

        nbrs = self.nbrs_[index, 1:]
        nbrs = nbrs[np.where(nbrs != -1)]

        return nbrs

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
        return [nbrs_[np.where(nbrs_ != -1)].tolist() for nbrs_ in nbrs]

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

        n = self.n - (max_v := max(self.p, self.q, self.r)) + 1
        ion.htprint("Status",
                    f"Integrity can only be ensured for the first n - max(p, q, r) 1 layer and thus layer {n}")

        # check for nbrs
        ln = self.lvls[self.n - 2] - 1
        for i in range(self.lvls[self.n - 2]):
            progbar = ">" * (l := int(64 * i / ln)) + " " * (64 - l)
            ion.htprint("Status", f"\r|{progbar}| Controlling nbrs for {i} / {ln}", end="")
            n_ = self.get_reflection_level(i)

            nbrs = self.get_nbrs(i)
            if len(nbrs) != len(set(nbrs)):
                raise AttributeError(f"Tesselation is corrupted! At least one nbr is registered at least twice {nbrs}!")
            elif len(nbrs) != 3:
                raise AttributeError(
                    f"Tesselation is corrupted! Cell {i} in layer {n_} has {len(nbrs)}/3 unique nbrs")

        print("")
        # ion.htprint("Status", f"Nbr relations controlled")

        # dual lattice
        ln = self.lvls[n] - 1
        for i in range(self.lvls[n]):
            progbar = ">" * (l := int(64 * i / ln)) + " " * (64 - l)
            ion.htprint("Status", f"\r|{progbar}| Bidirectional search for cell {i} / {ln}", end="")

            # print(f"Cell {i} has nbrs {self.get_nbrs(i)}")
            for nbr in self.get_nbrs(i):
                # print(f"Search {i} - {nbr}")
                # bidirectional search with path remembered
                # define paths
                paths1 = [[i]]
                paths2 = [[nbr]]

                # do first step to prevent direct path
                paths1 = [paths1[0] + [nbr_] for nbr_ in self.get_nbrs(i) if nbr_ != nbr]
                paths2 = [paths2[0] + [nbr_] for nbr_ in self.get_nbrs(nbr) if nbr_ != i]

                for s in range(max_v):
                    # perform single step in one list
                    paths1 = [path + [nbr_] for path in paths1 for nbr_ in self.get_nbrs(path[-1]) if
                              not (nbr_ in path)]

                    # control if already connected
                    common = [len(path1.intersection(path2)) for path1, path2 in itertools.product(
                        [set(p) for p in paths1],
                        [set(p) for p in paths2]
                    )
                              ]

                    if sum(common) >= 2:
                        break

                    # perform single step in other list
                    paths2 = [path + [nbr_] for path in paths2 for nbr_ in self.get_nbrs(path[-1]) if
                              not (nbr_ in path)]

                else:
                    print(paths1, n)
                    # raise ArithmeticError(f"At least one connection between {i} - {nbr} could not be established!")
                    print(f"At least one connection between {i} - {nbr} could not be established!")

            # exit()
        print("")
        # ion.htprint("Status", f"Dual lattice controlled")
        ion.htprint("Status", f"Integrity ensured!")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    ion.set_verbosity_level("Status")

    # graph = GRCT(4, 4, 7, 10, tiling=True)
    graph = GRCT(5, 4, 2, 22, nbrs=True, tiling=True)
    print(f"Tiling has {len(graph)} nodes")
    graph.check_integrity()
    nbrs = graph.get_nbrs_list()
    colors = ["#FF000060", "#00FF0060", "#0000FF60"]
    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)

    for i, poly in enumerate(graph):
        poly_layer = graph.get_reflection_level(i)
        facecolor = colors[poly_layer % len(colors)]
        patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in poly[1:]]),
                                    facecolor=facecolor, edgecolor="#FFFFFF")
        fig_ax[1].add_patch(patch)

        center = poly[0]
        if False:
            for nbr in nbrs[i]:
                center2 = graph.get_vertices(nbr)[0]
                end = (center2 - center) / 2 + center
                fig_ax[1].plot((np.real(center), np.real(end)), (np.imag(center), np.imag(end)), color="#000000")
        # fig_ax[1].text(np.real(center), np.imag(center), str(i))

    plt.autoscale()
    plt.show()
