import numpy as np
from typing import List
import math
from .distance import weierstrass_distance, lorentzian_distance
from .representations import p2w
from .ion import htprint

# This module contains neighbour search algorithms that can be applied to any tiling

def find_radius_brute_force(tiling, radius=None, eps=1e-5) -> List[List[int]]:
    """
    Get adjacent polygons for the entire tiling through radius search
    This algorithm works in a brute-force manner, the distances between 
    every pair of cells are compared against the search radius.

    Time complexity: O(n^2) where n=len(tiling)
    Slow, use only for small tilings or for debugging purposes

    Arguments:
    ----------
    tiling : sub-class of Tiling
        The hyperbolic tiling object (represented by one of the "kernels")
    radius : float
        The search radius
    eps : float
        Add small value to search radius to avoid rounding issues

    Returns:
    --------
        List[List[int]] containing neighbour indices of every cell.
    """
    if radius is None:
        htprint("Status", "No search radius provided; Assuming lattice spacing of the tessellation!")
        radius = tiling.h

    retlist = []  # prepare list

    for i in range(len(tiling)):
        sublist = []
        for j in range(len(tiling)):
            c1 = tiling.get_center(i)
            c2 = tiling.get_center(j)
            dist = weierstrass_distance(p2w(c1), p2w(c2))
            if tiling.h/2 <dist < radius + eps:
                sublist.append(j)

        retlist.append(sublist)
    return retlist


def find_radius_optimized(tiling, radius=None, eps=1e-5) -> List[List[int]]:
    """
    Get adjacent polygons for the entire tiling through radius search
    Compared to its brute-force equivalent, this improved implemention
    makes sure everything is fully vectorized and complied by numpy, 
    such that we gain a dramatic speed-up

    Time complexity: O(n^2) where n=len(tiling)

    Arguments:
    ----------
    tiling : sub-class of Tiling
        The hyperbolic tiling object (represented by one of the "kernels")
    radius : float
        The search radius
    eps : float
        Add small value to search radius to avoid rounding issues

    Returns:
    --------
        List[List[int]] containing neighbour indices of every cell.
    """

    if radius is None:
        htprint("Status", "No search radius provided; Assuming lattice spacing of the tessellation!")
        radius = tiling.h

    # prepare array containing all center coordinates
    # in Weierstrass representation
    ncells = len(tiling)
    v = np.zeros((ncells, 3))
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
        w = p2w(tiling.get_center(i)) # Weierstrass representation
        dists = lorentzian_distance(v, w)
        dists[(dists < 1)] = 1  # this costs some %, but reduces warnings
        indxs = np.where(dists < searchdist)[0]  # radius search
        selff = np.argwhere(indxs == i)  # find self
        indxs = np.delete(indxs, selff)  # delete self
        retlist.append(list(indxs))
    return retlist



def find_radius_optimized_single(tiling, index, radius=None, eps=1e-5) -> List[int]:
    """
    Get adjacent polygons for a single polygon through radius search
    Compared to its brute-force equivalent, this improved implemention
    makes sure everything is fully vectorized and complied by numpy, 
    such that we gain a dramatic speed-up

    Time complexity: O(n) where n=len(tiling)

    Arguments:
    ----------
    tiling : sub-class of Tiling
        The hyperbolic tiling object (represented by one of the "kernels")
    index : int
        Index of the cell for which the neighbours are to be found
    radius : float
        The search radius
    eps : float
        Add small value to search radius to avoid rounding issues

    Returns:
    --------
        [List[int]] containing neighbour indices of every cell.
    """

    if radius is None:
        htprint("Status", "No search radius provided; Assuming lattice spacing of the tessellation!")
        radius = tiling.h

    # prepare array containing all center coordinates
    # in Weierstrass representation
    ncells = len(tiling)
    v = np.zeros((ncells, 3))
    for i in range(ncells):
        v[i] = p2w(tiling.get_center(i))

    # add something to "radius" to avoid rounding problems
    # does not need to be particularly small
    searchdist = radius + eps
    searchdist = math.cosh(searchdist)

    w = p2w(tiling.get_center(index)) # Weierstrass representation
    dists = lorentzian_distance(v, w)
    dists[(dists < 1)] = 1  # this costs some %, but reduces warnings
    indxs = np.where(dists < searchdist)[0]  # radius search
    selff = np.argwhere(indxs == index)  # find self
    indxs = np.delete(indxs, selff)  # delete self
    return list(indxs)
