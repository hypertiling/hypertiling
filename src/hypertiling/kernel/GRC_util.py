from typing import Tuple
import numpy as np
from hypertiling.check_numba import NumbaChecker
import hypertiling.arraytransformation as array_trans

D = 0  # default = regular
L = 1  # left symmetric filler
F = 2  # asymmetric filler
R = 3  # right symmetric filler


# @NumbaChecker("void(complex128[:], complex128)")
# def tf(z, z0):
#    divi = (1 - z * np.conjugate(z0))
#    z -= z0
#    z /= divi

@NumbaChecker("int32[:](int32, int32, int32)")
def n2polyN(p, q, n):
    """
    Calculates the number of cells the tesselation requires.
    This method is similar to get_reflection_n_estimation in GR_util.py.
    However, as GR, GRG and GRGS are deprecated and will be moved to legacy in the future, this function was moved here

    Parameters
    ----------
    p : int
        Number of edges.
    q : int
        Number of polys per vertex.
    n : int
        Number of layers (reflective).

    Returns
    -------
    np.array[np.uint32]
        Number of tiles per layer.
    """
    lengths = np.empty((n,), dtype=np.int32)
    lengths[0] = 0

    if n == 1:
        return lengths

    lengths[1] = p

    if q == 3:
        k = p - 4
        for i in range(2, n):
            lengths[i] = k * lengths[i - 1] - lengths[i - 2]
        lengths[0] = 1
        return lengths

    fillers = np.zeros((n + n,), dtype=np.uint32)
    k = p - 1
    p2 = p - 2
    p3 = p - 3
    if q & 1:
        delta = (q - 1) // 2

        if delta < n:
            start = delta + 1
            fillers[delta + n] = p + p
            for i in range(2, start):
                lengths[i] = k * lengths[i - 1]
        else:
            start = 2

        for i in range(start, n):
            i_delta = i - delta
            i1 = i - 1

            fillers[i] = fillers[i_delta + n] // 2
            fillers[i + n] = 2 * (p2 * (lengths[i_delta] - fillers[i_delta] - fillers[i_delta + n]) +
                                  p3 * (fillers[i_delta] + fillers[i_delta + n]))
            lengths[i] = k * lengths[i1] - fillers[i1 + n] - fillers[i] - fillers[i1]

    else:
        delta = q // 2
        if delta < n:
            fillers[delta] = p

        for i in range(2, n):
            if i <= delta:
                lengths[i] = k * lengths[i - 1] - fillers[i]
            else:
                i_delta = i - delta
                i1 = i - 1
                fillers[i] = p2 * (lengths[i_delta] - fillers[i_delta]) + p3 * fillers[i_delta]
                lengths[i] = k * lengths[i1] - fillers[i] - fillers[i1]

    lengths[0] = 1
    return lengths


@NumbaChecker("void(int32, int32, int32, int32[:,:], int32[:], int32, int32, int32)")
def register(q: int, p_index: int, c_index: int, nbrs: np.array, types: np.array, mipi: int, mapi: int, mici: int):
    """
    Function for tracking the neighbor relations

    Parameters
    ----------
    q: int
        Number of cells per vertex
    p_index : int
       Index of the parent polygon
    c_index : int
       Index of the child to be created
    nbrs : np.array[int, int]
       Array yielding the nbr relations for all polygons
    types : np.array[int]
       Array yielding the type of each cell
    mipi : int
       MInimal Parent Index = index of the first cell in the same layer as parent
    mapi : int
       MAximal Parent Index = index of the last cell in the same layer as parent
    mici : int
       MInimal Child Index = index of the first cell in the same layer as the child
   """
    # connect parents
    nbrs[c_index, nbrs[c_index, 0]] = p_index
    nbrs[c_index, 0] += 1
    nbrs[p_index, nbrs[p_index, 0]] = c_index
    nbrs[p_index, 0] += 1

    # handle second kind fillers
    if (q == 3 or types[c_index] == R) and c_index != mici:
        ci1 = c_index - 1
        nbrs[c_index, nbrs[c_index, 0]] = ci1
        nbrs[c_index, 0] += 1
        nbrs[ci1, nbrs[ci1, 0]] = c_index
        nbrs[ci1, 0] += 1

    # handle first kind fillers
    if types[c_index] == F:
        pi1 = p_index + 1
        pi1 = pi1 if pi1 <= mapi else mipi
        nbrs[c_index, nbrs[c_index, 0]] = pi1
        nbrs[c_index, 0] += 1
        nbrs[pi1, nbrs[pi1, 0]] = c_index
        nbrs[pi1, 0] += 1


@NumbaChecker("boolean(int32, int32, int32, int32, int32[:], int32[:,:], int32[:,:], int32, int32, int32, int32)")
def propagate(q: int, p_index: int, c_index: int, edge: int, types: np.array, fillers: np.array, counters: np.array,
              lc: int, delta: int, starttype: int, fillerstarttype: int) -> bool:
    """
    Function performing the cell propagation (including the cell types)

    Parameters
    ----------
    q : int
        Number of cells per vertex
    p_index : int
        Index of the parent polygon
    c_index : int
        Index of the child to be created
    edge : int
        Edge index the reflection is performed on
    types : np.array[int]
       Array yielding the type of each cell
    fillers : np.array[int, int]
        Array yielding the expected fillers for the first and last edge for each cell
    counters : np.array[int, int]
        Array yielding the counters for first and last edge for each cell
    lc : int
        Last child = Number indicating the number of childs the cell can have maximally
    delta : int
        Number of layers involved in closing a vertex once it was opened (q - 1) // 2
    starttype : int
        Type newly created cells will have by default
    fillerstarttype : int
        Type a newly opened vertex will have by default

    Returns
    -------
    bool
        Indicates if a child was created or if the edge was blocked
    """
    fc = 0

    # handle q == 3
    if q == 3:
        if types[p_index] == F:
            lc -= 1
        fc += 1

    # handle special cases
    if types[p_index] == R:
        if edge == fc:
            return False
        fc += 1
    elif types[p_index] == L or types[p_index] == F:
        if edge == lc:
            return False
        lc -= 1

    if edge == lc:
        counters[c_index, 0] = counters[p_index, 0] - 1
        fillers[c_index, 0] = fillers[p_index, 0]
    else:
        counters[c_index, 0] = delta - 1
        fillers[c_index, 0] = fillerstarttype

    if edge == fc:
        counters[c_index, 1] = counters[p_index, 1] - 1
        fillers[c_index, 1] = fillers[p_index, 1]
    else:
        counters[c_index, 1] = delta - 1
        fillers[c_index, 1] = fillerstarttype

    if counters[c_index, 0] == 0:
        types[c_index] = fillers[p_index, 0]
        counters[c_index, 0] = delta - 1

        if types[c_index] == F:
            fillers[c_index, 0] = fillerstarttype
        else:
            types[c_index] = L
            fillers[c_index, 0] = F
            counters[c_index, 0] = delta

    elif counters[c_index, 1] == 0:
        types[c_index] = fillers[p_index, 1]
        if types[c_index] == F:
            return False
        types[c_index] = R
        fillers[c_index, 1] = F
        counters[c_index, 1] = delta
    else:
        types[c_index] = starttype

    return True


@NumbaChecker("void(int32, int32, int32, int32, complex128[:,:])")
def propagate_coords(p: int, p_index: int, c_index: int, edge: int, coords: np.array):
    """
    Function performing the propagation of coordinates (tiling=True)

    Parameters
    ----------
    p : int
        Number of edges per polygon
    p_index : int
        Index of the parent polygon
    c_index : int
        Index of the child to be created
    edge : int
        Edge index the reflection is performed on
    coords : np.array[int, int]
        Array yielding the coordinates of all triangles
    """
    coords[c_index] = coords[p_index]
    z0 = coords[c_index, edge + 1]

    # tf(coords[c_index], z0)
    array_trans.morigin(p, z0, coords[c_index])

    phi2 = 2 * np.angle(coords[c_index, (edge + 1) % p + 1])  # phi2 = phi + phi
    coords[c_index] = np.conjugate(coords[c_index])
    coords[c_index] *= complex(np.cos(phi2), np.sin(phi2))

    # tf(coords[c_index], - z0)
    array_trans.morigin(p, -z0, coords[c_index])

    coords[c_index, 1:] = np.roll(np.flip(coords[c_index, 1:]), edge + 1)


@NumbaChecker("Tuple((complex128[:,:], int32[:,:], int32[:]))(int32, int32, int32, boolean, boolean)")
def construct_sector(p: int, q: int, n: int, tiling: bool, nbrs_: bool) -> Tuple[np.array, np.array, np.array]:
    """
    Construct the tesselation for the fundamental sector

    Parameters
    ----------
    p : int
        Number of edges per cell
    q : int
        Number of cells meeting at a single vertex
    n : int
        Number of layers
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
    # calculate length of tiling
    lengths = n2polyN(p, q, n)
    lengths //= p
    lengths[0] = 1
    length = np.sum(lengths)

    # prepare everything necessary for even n = 1
    lvls = np.empty((n,), dtype=np.int32)
    lvls[0] = 1

    if nbrs_:
        nbrs = np.full((length, p + 1), -1, dtype=np.int32)
        nbrs[:, 0] = 1
    else:
        nbrs = np.empty((1, 1), dtype=np.int32)

    if tiling:
        coords = np.empty((length, p + 1), dtype=np.cdouble)
        pq = p * q
        r = np.sqrt(np.cos(np.pi * (p + q) / pq) / np.cos(np.pi * (p - q) / pq))
        coords[0, 0] = 0
        coords[0, 1:] = r * np.exp(2j * np.pi * np.arange(p) / p)
    else:
        coords = np.empty((1, 1), dtype=np.cdouble)

    if n == 1:
        return coords, nbrs, lvls

    types = np.empty((length,), dtype=np.int32)  # D:0, L:1, F:2, S:3
    counters = np.empty((length, 2), dtype=np.int32)  # lc, rc
    fillers = np.empty((length, 2), dtype=np.int32)  # lt, rt

    # determine some helpful constants
    k = p - 1
    k1 = k - 1
    delta = q // 2
    starttype = D if q != 3 else L
    fillerstarttype = 3 if q & 1 else 2
    if q == 3:
        delta += 1
        fillerstarttype = F

    # prepare array helpers
    csi = 1
    cei = 2
    polycounter = 2

    # create fundamental polygon
    types[0] = 0
    counters[0] = delta
    fillers[0] = fillerstarttype

    # calculate first layer
    propagate(q, 0, 1, 0, types, fillers, counters, k1, delta, starttype, fillerstarttype)
    counters[1] = delta - 1

    if nbrs_:
        # handle nbr relations
        nbrs[1, 1] = 0
        nbrs[1, 0] += 1
        nbrs[0, 1] = 1
        nbrs[0, 0] += 1

    if tiling:
        propagate_coords(p, 0, 1, 0, coords)

    if nbrs_ and q == 3:
        nbrs[1, 2] = length
        nbrs[1, 3] = (p - 1) * (length - 1) + 1
        nbrs[1, 0] += 2

    lvls[1] = 2
    if n == 2:
        return coords, nbrs, lvls

    # all other layers
    l1 = length - 1
    p1l1 = (p - 1) * l1
    edge_k = k if q != 3 else k1
    edge_start = 0 if q != 3 else 1
    for lvl in range(2, n):
        cei1 = cei - 1
        for p_index in range(csi, cei):
            for edge in range(edge_start, edge_k):
                if propagate(q, p_index, polycounter, edge, types, fillers, counters, k1, delta, starttype,
                             fillerstarttype):
                    if nbrs_:
                        register(q, p_index, polycounter, nbrs, types, csi, cei1, cei)
                    if tiling:
                        propagate_coords(p, p_index, polycounter, edge, coords)
                    polycounter += 1

            # first kind filler
        lci = polycounter - 1
        if nbrs_ and types[lci] == F:
            nbrs[csi, nbrs[csi, 0] - 1] = lci + p1l1
            nbrs[lci, nbrs[lci, 0] - 1] = csi + l1

        # second kind filler
        if nbrs_ and (types[cei] == R or q == 3):
            nbrs[cei, nbrs[cei, 0]] = lci + p1l1
            nbrs[cei, 0] += 1
            nbrs[lci, nbrs[lci, 0]] = cei + l1
            nbrs[lci, 0] += 1

        csi = cei
        cei = polycounter
        lvls[lvl] = polycounter

    return coords, nbrs, lvls


@NumbaChecker("Tuple((complex128[:,:], int32[:,:], int32[:]))(int32, int32, int32, boolean, boolean)")
def construct_full(p, q, n, tiling, nbrs_):
    """
    Construct the tesselation (fully)

    Parameters
    ----------
    p : int
        Number of edges per cell
    q : int
        Number of cells meeting at a single vertex
    n : int
        Number of layers
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
    # calculate length of tiling
    lengths = n2polyN(p, q, n)
    length = np.sum(lengths)

    # prepare everything necessary for even n = 1
    lvls = np.empty((n,), dtype=np.int32)
    lvls[0] = 1

    if nbrs_:
        nbrs = np.full((length, p + 1), -1, dtype=np.int32)
        nbrs[:, 0] = 1
    else:
        nbrs = np.empty((1, 1), dtype=np.int32)

    if tiling:
        coords = np.empty((length, p + 1), dtype=np.cdouble)
        pq = p * q
        r = np.sqrt(np.cos(np.pi * (p + q) / pq) / np.cos(np.pi * (p - q) / pq))
        coords[0, 0] = 0
        coords[0, 1:] = r * np.exp(2j * np.pi * np.arange(p) / p)
    else:
        coords = np.empty((1, 1), dtype=np.cdouble)

    if n == 1:
        return coords, nbrs, lvls

    types = np.empty((length,), dtype=np.int32)  # D:0, L:1, F:2, S:3
    counters = np.empty((length, 2), dtype=np.int32)  # lc, rc
    fillers = np.empty((length, 2), dtype=np.int32)  # lt, rt

    # determine some helpful constants
    k = p - 1
    k1 = k - 1
    delta = q // 2
    starttype = D if q != 3 else L
    fillerstarttype = 3 if q & 1 else 2
    if q == 3:
        delta += 1
        fillerstarttype = F

    # prepare array helpers
    csi = 1
    cei = p + 1
    polycounter = 1

    # create fundamental polygon
    types[0] = 0
    counters[0] = delta
    fillers[0] = fillerstarttype

    # calculate first layer
    for i in range(p):
        propagate(q, 0, polycounter, i, types, fillers, counters, k1, delta, starttype, fillerstarttype)
        counters[polycounter] = delta - 1
        if nbrs_:
            register(q, 0, polycounter, nbrs, types, 0, 0, 1)
        if tiling:
            propagate_coords(p, 0, polycounter, i, coords)

        polycounter += 1

    if nbrs_ and q == 3:
        nbrs[1, nbrs[1, 0]] = p
        nbrs[1, 0] += 1
        nbrs[p, nbrs[p, 0]] = 1
        nbrs[p, 0] += 1

    lvls[1] = polycounter

    if n == 2:
        return coords, nbrs, lvls

    # all other layers
    edge_k = k if q != 3 else k1
    edge_start = 0 if q != 3 else 1
    for lvl in range(2, n):
        cei1 = cei - 1
        for p_index in range(csi, cei):
            for edge in range(edge_start, edge_k):
                if propagate(q, p_index, polycounter, edge, types, fillers, counters, k1, delta, starttype,
                             fillerstarttype):
                    if nbrs_:
                        register(q, p_index, polycounter, nbrs, types, csi, cei1, cei)
                    if tiling:
                        propagate_coords(p, p_index, polycounter, edge, coords)
                    polycounter += 1

        # second kind filler
        if types[cei] == R or q == 3:
            lci = polycounter - 1
            if nbrs_:
                nbrs[cei, nbrs[cei, 0]] = lci
                nbrs[cei, 0] += 1
                nbrs[lci, nbrs[lci, 0]] = cei
                nbrs[lci, 0] += 1

        csi = cei
        cei = polycounter
        lvls[lvl] = polycounter

    return coords, nbrs, lvls


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import time

    p, q, n = 3, 7, 4

    t1 = time.time()
    coords, nbrs, lvls = construct_full(p, q, n, True, True)
    t2 = time.time()
    print(f"Took {t2 - t1}s with a total of {lvls[-1]} triangles")
    print(lvls)
    # exit()

    colors = ["#FF000060", "#00FF0060", "#0000FF60"]
    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)


    def get_reflection_level(index: int) -> int:
        level = np.searchsorted(lvls, index)
        if lvls[level] == index:
            level += 1
        return level


    for i in range(0, coords.shape[0]):
        facecolor = colors[get_reflection_level(i) % len(colors)]
        patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in coords[i, 1:]]), facecolor=facecolor,
                                    edgecolor="#FFFFFF")
        fig_ax[1].add_patch(patch)

        center = coords[i][0]  # / p
        for nbr in nbrs[i, 1:]:

            if nbr == -1 or nbr >= lvls[-1]:
                continue

            center2 = coords[nbr][0]
            end = (center2 - center) / 2 + center
            fig_ax[1].plot((np.real(center), np.real(end)), (np.imag(center), np.imag(end)), color="#000000")

        fig_ax[1].text(np.real(center), np.imag(center), str(i))

    plt.show()
