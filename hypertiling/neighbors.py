import numpy as np
from typing import Callable, Any, List
import math
from hypertiling.distance import weierstrass_distance, lorentzian_distance
from hypertiling.util import lattice_spacing_weierstrass
from hypertiling.transformation import p2w




# wrapper to provide a nicer interface
def find(tiling, nn_dist=None, which="optimized", index_from_zero=True, verbose=False):
    if nn_dist == None:
        print("[hypertiling] No search radius given;\
            Assuming lattice spacing of the tessellation!")
        nn_dist = lattice_spacing_weierstrass(self.p, self.q)

    if which == "optimized_slice":
        retval = find_nn_optimized_slice(tiling, nn_dist)  # fastest
    elif which == "optimized":
        retval = find_nn_optimized(tiling, nn_dist)
    elif which == "brute_force":
        retval = find_nn_brute_force(tiling, nn_dist)  # use for debug
    elif which == "slice":
        retval = find_nn_slice(tiling, nn_dist)
    elif which == "edge_map":
        retval = find_nn_edge_map_optimized(tiling)
    elif which == "edge_map_brute_force":
        retval = find_nn_edge_map_brute_force(tiling)

    else:
        print("[Hypertiling] Error:", which, " is not a valid algorithm!")

    nbrs = []
    if index_from_zero:
        for sublist in retval:
            new_sublist = [x - 1 for x in sublist]
            nbrs.append(new_sublist)

        return nbrs
    else:
        return retval


def find_bfr(tiling, radius: float, eps=1e-5) -> List[List[int]]:
    """
    Get adjacent polygons for the entire tiling through radius search
    This algorithm works in a brute-force manner, the distances between 
    every pair of cells are compared against the search radius.

    Time complexity: O(n^2) where n=len(tiling)
    Slow, use only for small tilings or for debugging purposes

    Arguments:
    ----------
    tiling : sub-class of AbstractKernelBase
        The hyperbolic tiling object (represented by one of the "kernels")
    radius : float
        The search radius
    eps : float
        Add small value to search radius to avoid rounding issues

    Returns:
    --------
        List[List[int]] containing neighbour indices of every cell.
    """

    retlist = []  # prepare list


    for i in range(len(tiling)):
        sublist = []
        for j in range(len(tiling)):
            c1 = tiling.get_center(i)
            c2 = tiling.get_center(j)
            dist = weierstrass_distance(p2w(c1), p2w(c2))
            if dist < radius + eps:
                if i is not j:
                    sublist.append(j)
        retlist.append(sublist)
    return retlist



def find_ro(tiling, radius, eps=1e-5):
    """
    Get adjacent polygons for the entire tiling through radius search
    Compared to its brute-force equivalent, this improved implemention
    makes sure everything is fully vectorized by numpy, such that we
    gain a significant speed-up

    Time complexity: O(n^2) where n=len(tiling)

    Arguments:
    ----------
    tiling : sub-class of AbstractKernelBase
        The hyperbolic tiling object (represented by one of the "kernels")
    radius : float
        The search radius
    eps : float
        Add small value to search radius to avoid rounding issues

    Returns:
    --------
        List[List[int]] containing neighbour indices of every cell.
    """



    # prepare matrix containing all center coordinates
    ncells = len(tiling)
    v = np.zeros(ncells, 3))
    for i in range(ncells):
        v[i] = p2w(tiling.get_center(i))

    # add something to "radius" to avoid rounding problems
    # does not need to be particularly small
    searchdist = radius + eps
    searchdist = math.cosh(searchdist)

    # prepare list
    retlist = []

    # loop over cells
    for i in range(ncells):
        w = p2w(tiling.get_center(i))
        dists = lorentzian_distance(v, w)
        dists[(dists < 1)] = 1  # this costs some %, but reduces warnings
        indxs = np.where(dists < searchdist)[0]  # radius search
        selff = np.argwhere(indxs == i)  # find self
        indxs = np.delete(indxs, selff)  # delete self
        retlist.append(list(indxs))
    return retlist



