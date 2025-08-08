from typing import Tuple
from hypertiling.check_numba import NumbaChecker
import numpy as np
import hypertiling.arraytransformation as array_trans


# @NumbaChecker("complex128[:](complex128[:], complex128)")
# def tf(z, z0):
#   return (z - z0) / (1 - z * np.conjugate(z0))


@NumbaChecker("float64(int32, int32, int32)")
def distance(p: int, q: int, r: int) -> float:
    """
    Calculates the spartial extension of the fundamental triangle.

    Time-complexity: O(1)

    Parameters
    ----------
    p : int
        Number of cells on the 1st vertex
    q : int
        Number of cells on the 2nd vertex
    r : int
        Number of cells on the 3rd vertex

    Returns
    -------
    float
        Spatial length of the triangle in that direction
    """
    return (np.cos(np.pi / p) + np.cos(np.pi / q) * np.cos(np.pi / r)) / (np.sin(np.pi / q) * np.sin(np.pi / r))


@NumbaChecker("int32(int32, int32, int32, int32)")
def get_n(p: int, q: int, r: int, n: int) -> int:
    """
    Very rough estimation of the number of cells the tesselation will have.
    We are working on getting a formula here!

    Parameters
    ----------
    p : int
        Number of cells on the 1st vertex
    q : int
        Number of cells on the 2nd vertex
    r : int
        Number of cells on the 3rd vertex
    n : int
        Number of layers

    Returns
    -------
    np.array[np.uint32]
        Number of cells for the tesselation
    """
    numbers = 2 * r * 2 ** n
    return numbers


@NumbaChecker("boolean(int32, int32, int32, int32, int32[:,:], int32[:,:], int32[:,:], boolean)")
def propagate(p_index: int, cp_index: int, c_index: int, epv: int, edges: np.array, counters: np.array, flags: np.array,
              enforce: bool) -> bool:
    """
    Function performing the cell propagation (including the cell types)

    Parameters
    ----------
    p_index : int
        Index of the parent polygon
    cp_index : int
        Index of a possible coparent
    c_index : int
        Index of the child to be created
    epv : int
        Edge index the reflection is performed on
    edges : np.array[int, int]
        Array yielding the permutations of [p, q, r] for all cells
    counters : np.array[int, int]
        Array yielding the counters for each edge for each cell
    flags : np.array[int, int]
        Array yielding the flags (types) for each edge for each cell
    enforce : bool
        Determines if the propagation is enforced regardless of type and other states

    Returns
    -------
    bool
        Indicates if a child was created or if the edge was blocked
    """
    # -1 for previous edge, + 3 for triangle (to compensate for 3) => + 2
    apv = (epv + 2) % 3
    npv = epv + 1  # (epv + 1) % 3

    # check if edge is allowed
    if (counters[p_index, epv] == 0 or counters[p_index, apv] == 0) and not enforce:
        return False

    # copy properties
    counters[c_index] = counters[p_index]
    flags[c_index] = flags[p_index]

    # update counters & flags
    flags[c_index, npv] = 0
    counters[c_index, epv] -= 1
    counters[c_index, apv] -= 1
    counters[c_index, npv] = edges[p_index, npv]

    # first kind triangles
    if flags[c_index, epv] == 1:
        counters[c_index, epv] -= 1
        flags[c_index, epv] = 2

        # if np.sum(counters[cp_index] == 0) == 2:
        # if np.count_nonzero(counters[cp_index] == 0) == 2:
        # (counters[cp_index, 0] == 0 and counters[cp_index, 1] == 0) or \
        #                 (counters[cp_index, 0] == 0 and counters[cp_index, 2] == 0) or \
        if counters[cp_index, 1] == 0 and counters[cp_index, 2] == 0:
            counters[c_index, npv] = counters[cp_index, 0]

        if flags[cp_index, epv] == 1 and flags[cp_index, npv] == 2:
            flags[c_index, npv] = 1

    elif flags[c_index, apv] == 1 and counters[c_index, apv] == 0:
        flags[c_index, npv] = 1

    # second kind triangles
    if (counters[c_index, epv] == 0 and flags[c_index, epv] != 2) or \
            (counters[c_index, apv] == 0 and flags[c_index, apv] != 2):
        counters[c_index, npv] -= 1

    # regular triangles
    # if np.all((counters[c_index] + (flags[c_index] == 2)) > 0):
    if (counters[c_index, 0] > 0 or flags[c_index, 0] == 2) and \
            (counters[c_index, 1] > 0 or flags[c_index, 1] == 2) and \
            (counters[c_index, 2] > 0 or flags[c_index, 2] == 2):
        flags[c_index, npv] = 1

    # correct orientation
    flags[c_index] = np.flip(flags[c_index])
    edges[c_index] = np.flip(edges[p_index])
    counters[c_index] = np.flip(counters[c_index])

    if epv == 0:
        buffer = flags[c_index, 0]
        flags[c_index, :2] = flags[c_index, 1:]
        flags[c_index, 2] = buffer

        buffer = edges[c_index, 0]
        edges[c_index, :2] = edges[c_index, 1:]
        edges[c_index, 2] = buffer

        buffer = counters[c_index, 0]
        counters[c_index, :2] = counters[c_index, 1:]
        counters[c_index, 2] = buffer
        # flags[c_index] = np.roll(flags[c_index], epv2)
        # edges[c_index] = np.roll(edges[c_index], epv2)
        # counters[c_index] = np.roll(counters[c_index], epv2)

    # print(f"{p_index} > {c_index}:{counters[c_index]}-{flags[c_index]}")
    return True


@NumbaChecker("void(int32, int32, int32, complex128[:,:])")
def propagate_coords(p_index: int, c_index: int, epv: int, coords: np.array):
    """
    Function performing the propagation of coordinates (tiling=True)

    Parameters
    ----------
    p_index : int
        Index of the parent polygon
    c_index : int
        Index of the child to be created
    epv : int
        Edge index the reflection is performed on
    coords : np.array[int, int]
        Array yielding the coordinates of all triangles
    """
    apv = (epv + 2) % 3

    coords[c_index] = coords[p_index]
    z0 = coords[c_index, epv + 1]

    # coords[c_index] = tf(coords[c_index], z0)
    array_trans.morigin(3, z0, coords[c_index])

    phi2 = 2 * np.angle(coords[c_index, apv + 1])
    coords[c_index] = np.conjugate(coords[c_index])
    coords[c_index] *= complex(np.cos(phi2), np.sin(phi2))

    # coords[c_index] = tf(coords[c_index], - z0)
    array_trans.morigin(3, -z0, coords[c_index])

    coords[c_index, 1:] = np.flip(coords[c_index, 1:].copy())  # copy necessary for numba

    # substitutes roll for triangles
    if epv == 0:
        # coords = [c, p, q, r]
        # buffer = p
        # coords = [c, q, r, r]
        # coords = [c, q, r, p]
        buffer = coords[c_index, 1]
        coords[c_index, 1:3] = coords[c_index, 2:]
        coords[c_index, 3] = buffer


@NumbaChecker(["int32(int32, int32, int32[:,:], int32[:,:], int32[:,:], int32, boolean)"])
def register(p_index: int, c_index: int, nbrs: np.array, counters: np.array, flags: np.array, p_stop: int,
             block_filler: bool) -> int:
    """
    Function for tracking the neighbor relations

    Parameters
    ----------
    p_index : int
        Index of the parent polygon
    c_index : int
        Index of the child to be created
    nbrs : np.array[int, int]
        Array yielding the nbr relations for all polygons
    counters : np.array[int, int]
        Array yielding the counters for each edge for each cell
    flags : np.array[int, int]
        Array yielding the flags (types) for each edge for each cell
    p_stop : int
        Index of the first cell in the same layer as the child
    block_filler : bool
        Determines if the propagation is enforced regardless of type and other states

    Returns
    -------
    int
        State if all connections could be established already
        If 1, a partner polygon was not yet created and the connection is established after the creation of the layer is
        completed (Automatically done in construct function)
        If 0, all neighbor relations where established successfully
    """

    # parent - child relation
    nbrs[c_index, nbrs[c_index, 0]] = p_index
    nbrs[p_index, nbrs[p_index, 0]] = c_index
    nbrs[c_index, 0] += 1
    nbrs[p_index, 0] += 1

    if block_filler:
        return 0

    # second kind filler
    if counters[c_index, 1] == 0 and flags[c_index, 1] != 2:
        c1_index = c_index - 1
        # boundary filler
        if c1_index < p_stop:
            return 1

        nbrs[c_index, nbrs[c_index, 0]] = c1_index
        nbrs[c1_index, nbrs[c1_index, 0]] = c_index
        nbrs[c_index, 0] += 1
        nbrs[c1_index, 0] += 1
        return 0

    # first kind fillers
    for j in range(3):
        if counters[c_index, j] == 0 and flags[c_index, j] == 1:
            p1_index = p_index + 1
            nbrs[c_index, nbrs[c_index, 0]] = p1_index
            nbrs[p1_index, nbrs[p1_index, 0]] = c_index
            nbrs[c_index, 0] += 1
            nbrs[p1_index, 0] += 1
            return 0

    return 0


@NumbaChecker("Tuple((complex128[:,:], int32[:,:], int32[:]))(int32, int32, int32, int32, int32, boolean, boolean)")
def construct_full(p: int, q: int, r: int, n: int, size: int, tiling: bool, nbrs_: bool) -> Tuple[
    np.array, np.array, np.array]:
    """
    Construct the full tesselation

    Parameters
    ----------
    p : int
        Number of cells on the 1st vertex
    q : int
        Number of cells on the 2nd vertex
    r : int
        Number of cells on the 3rd vertex
    n : int
        Number of layers
    size : int
        Number of nodes in the lattice (maximal)
    tiling : bool
        Indicates if coordinates should be calculated
    nbrs_ : bool
        Indicates if neighbor relations should be tracked

    Returns
    -------
    np.array[complex128]
        Array of coordinates for all polyongs
    np.array[int]
        Array of neighbor relations for all cells
    np.array[int]
        Array of layer sizes
    """

    # graph properties
    if nbrs_:
        nbrs = np.full((size, 4), -1, dtype=np.int32)  # [#nbrs, first, sec, third]
        nbrs[:, 0] = 1
    else:
        nbrs = np.empty((1, 1), dtype=np.int32)

    if tiling:
        coords = np.empty((size, 4), dtype=np.complex128)
    else:
        coords = np.empty((1, 1), dtype=np.complex128)

    edges = np.empty((size, 3), dtype=np.int32)
    counters = np.empty((size, 3), dtype=np.int32)
    flags = np.zeros((size, 3), dtype=np.int32)  # 1 = F, 2 = f
    lvls = np.empty(n + 1, dtype=np.int32)
    lvls[0] = 0
    lvls[1] = r + r

    if tiling:
        # create fundamental triangle
        l = distance(p, q, r)
        m = distance(q, p, r)
        coords[0, 1] = np.sqrt((l - 1) / (l + 1))
        coords[0, 2] = np.sqrt((m - 1) / (m + 1)) * np.exp(1j * np.pi / r)
        coords[0, 3] = 0
        coords[0, 0] = np.sum(coords[0, 1:]) / 3

    edges[0, 0] = q
    edges[0, 1] = p
    edges[0, 2] = r
    counters[0] = edges[0] - 1
    counters[0, 2] = 0

    poly_counter = 1

    # close vertex at r
    r2 = r + r
    for i in range(1, r2):
        propagate(poly_counter - 1, 0, poly_counter, 0, edges, counters, flags, True)
        if tiling:
            propagate_coords(poly_counter - 1, poly_counter, 0, coords)
        counters[poly_counter] = edges[poly_counter] - 1
        counters[poly_counter, 2] = 0

        if nbrs_:
            register(poly_counter - 1, poly_counter, nbrs, counters, flags, 1, True)
        poly_counter += 1

    if nbrs_:
        register(0, poly_counter - 1, nbrs, counters, flags, 1, True)
    counters[:poly_counter, 2] = 0

    if n == 1:
        return coords[:poly_counter], nbrs[:poly_counter], lvls

    # create first layer around to have set of polygons with only one edge parent
    skip_connection = 0
    for i in range(0, r2):
        propagate(i, 0, poly_counter, 1, edges, counters, flags, False)
        if tiling:
            propagate_coords(i, poly_counter, 1, coords)

        if nbrs_:
            skip_connection |= register(i, poly_counter, nbrs, counters, flags, r2, False)
        poly_counter += 1

    if nbrs_ and skip_connection == 1:
        pc1 = poly_counter - 1
        nbrs[r2, nbrs[r2, 0]] = pc1
        nbrs[pc1, nbrs[pc1, 0]] = r2
        nbrs[r2, 0] += 1
        nbrs[pc1, 0] += 1

    lvls[2] = poly_counter

    # create remaining tiling
    p_start = r2
    p_stop = poly_counter
    for i in range(n - 2):
        skip_connection = 0
        for poly_index in range(p_start, p_stop):
            cp_index = poly_index + 1
            cp_index = cp_index if cp_index < p_stop else p_start
            for edge in range(1, -1, -1):
                ans = propagate(poly_index, cp_index, poly_counter, edge, edges, counters, flags, False)
                if ans:
                    if tiling:
                        propagate_coords(poly_index, poly_counter, edge, coords)

                    if nbrs_:
                        skip_connection += register(poly_index, poly_counter, nbrs, counters, flags, p_stop, False)
                    poly_counter += 1

        if skip_connection == 1:
            pc1 = poly_counter - 1
            if nbrs_:
                nbrs[p_stop, nbrs[p_stop, 0]] = pc1
                nbrs[pc1, nbrs[pc1, 0]] = p_stop
                nbrs[p_stop, 0] += 1
                nbrs[pc1, 0] += 1

        lvls[3 + i] = poly_counter
        p_start = p_stop
        p_stop = poly_counter

    return coords[:poly_counter], nbrs[:poly_counter], lvls


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import time

    # p, q, r, n = 2, 3, 7, 12  # 15
    p, q, r, n = 5, 4, 2, 10  # 7, 11, 4
    # p, q, r, n = 8, 3, 2, 18

    # print("Start")
    t1 = time.time()
    coords, nbrs, lvls = construct_full(p, q, r, n, True, True)
    print(f"Took {time.time() - t1} s with a total of {lvls[-1]} triangles")


    def get_reflection_level(index: int) -> int:
        level = np.searchsorted(lvls, index)
        if lvls[level] == index:
            level += 1
        return level


    colors = ["#FF000060", "#00FF0060", "#0000FF60"]
    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)

    highlight = [1, 2, 5, 6, 11, 12, 21, 22, 35, 36] + [0, 4, 8, 9, 17]
    numbers = {i: e for i, e, in zip(highlight, [4, 4, 3, 3, 2, 2, 1, 1, 0, 0] + [False, 2, 1, 0, 0])}

    for i in range(0, coords.shape[0]):
        if i in highlight:
            facecolor = colors[get_reflection_level(i) % len(colors)]
            center = coords[i, 0]  # np.sum(coords[i]) / 3 - 0.02  # Magic number 0.02
            if not (numbers[i] is False):
                fig_ax[1].text(np.real(center), np.imag(center), str(numbers[i]))
        else:
            facecolor = "#AAAAAA60"

        patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in coords[i, 1:]]), facecolor=facecolor,
                                    edgecolor="#FFFFFF")
        fig_ax[1].add_patch(patch)

        # if False and not (i in []):
        #    continue

        # fig_ax[1].text(np.real(center), np.imag(center), str(i))
        # plt.plot((np.real(center), np.real(coords[i, 0])), (np.imag(center), np.imag(coords[i, 0])), "#000000")
        # for nbr in nbrs[i, 1:]:
        #    if nbr == -1:
        #        continue
        #    center2 = np.sum(coords[nbr]) / 3
        #    end = (center2 - center) / 2 + center
        #    fig_ax[1].plot((np.real(center), np.real(end)), (np.imag(center), np.imag(end)), color="#000000")

    plt.scatter(0.27545, 0.24072, s=50, marker="*", color="#FF0000")
    plt.scatter(0, -0.3035, s=50, marker=".", color="#FF0000")
    plt.axis('off')
    plt.show()
