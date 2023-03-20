from typing import Tuple
import numpy as np
import hypertiling.arraytransformation as array_trans
import hypertiling.transformation as trans
from hypertiling.check_numba import NumbaChecker

"""
p: Number of edges/vertices of a polygon
q: Number of polygons that meet at a vertex
n: Number of layers (classical definition)
m: Number of polygons
"""

# Variables ============================================================================================================


PI2 = 2 * np.pi


# Variables ============================================================================================================
# Assistance ===========================================================================================================


@NumbaChecker("boolean(complex128[::1], complex128, float64)")
def any_is_close(zs: np.array, z: np.complex128, tol: float) -> bool:
    """
    Compares if the complex z is in the array zs, with tolerance tol
    Time-complexity: O(p)
    :param zs: np.array[complex] = array with p complex to compare
    :param z: complex = the value to search for
    :param tol: float = tolerance of the comparison (absolut)
    :result: bool = True if float is in array else False
    """
    return np.any(np.abs(zs - z) <= tol)


@NumbaChecker(["boolean(complex128, complex128, float64)",
               "boolean(float64, float64, float64)",
               "boolean[::1](float64[::1], float64, float64)"])
def is_close_within_tol(z1: complex, z2: complex, tol: float) -> bool:
    """
    Compares if the complex z1 is equal to z2 up to tol
    Time-complexity: O(1)
    :param z1: Union[complex, float, int] = first value
    :param z2: Union[complex, float, int] = second value
    :param tol: float = tolerance of the comparison (absolut)
    :result: bool = True if both are equal up to tol
    """
    return np.abs(z1 - z2) <= tol


@NumbaChecker(["boolean(complex128, complex128)",
               "boolean(float64, float64)",
               "boolean[::1](float64[::1], float64)"])
def is_close(z1: complex, z2: complex) -> bool:
    return is_close_within_tol(z1, z2, 1E-12)


@NumbaChecker(["int64[:, :](complex128[::1], complex128[::1], float64)"])
def any_close_matrix_within_tol(zs1: np.array, zs2: np.array, tol: float) -> np.array:
    """
    Returns which points of zs1 and zs2 are closer (equal) to tol.
    Time-complexity: O(pq)
    :param zs1: np.array[complex] = array with p complex to compare
    :param zs2: np.array[complex] = array with q complex to compare
    :param tol: float = tolerance of the comparison (absolut)
    :result: np.array = positions where the points match
    """
    return np.argwhere(np.abs(zs1 - zs2.reshape(zs2.shape[0], 1)) <= tol)


@NumbaChecker(["int64[:, :](complex128[::1], complex128[::1])"])
def any_close_matrix(zs1: np.array, zs2: np.array) -> np.array:
    return any_close_matrix_within_tol(zs1, zs2, 1E-12)


@NumbaChecker("complex128[::1](complex128[::1])")
def generate_raw(poly: np.array) -> np.array:
    """
    Generates the neigboring polygons for a single polygon poly
    Time-complexity: O(p^2)
    :param poly: np.array[np.complex128][p + 1] = polygon to grow with p vertices
    :return: np.array[np.complex128][p] = centers of the neigboring polygons
    """
    reflection_centers = np.empty((poly.shape[0] - 1,), dtype=np.complex128)
    for k, vertex in enumerate(poly[1:]):
        z = poly.copy()
        array_trans.morigin(z.shape[0] - 1, vertex, z)
        phi = np.angle(z[1:][(k + 1) % (z.shape[0] - 1)])

        # from here: only use the center point
        z = trans.moeb_rotate_trafo(-phi, z[0])
        z = np.conjugate(z)
        z = trans.moeb_rotate_trafo(phi, z)
        z = trans.moeb_origin_trafo(- vertex, z)

        reflection_centers[k] = z
    return reflection_centers


@NumbaChecker("float64(complex128, complex128)")
def f_dist_disc(z: np.complex128, z_hat: np.complex128) -> float:
    """
    Calculates the distance between the points z and z_hat.
    Time-complexity: O(1)
    :param z: np.complex128 = first point
    :param z_hat: np.complex128 = second point
    :return: float = distance on disk
    """
    return 2 * np.arctanh(np.abs(z - z_hat) / np.abs(1 - z * z_hat.conjugate()))


# Assistance ===========================================================================================================
# Methods ==============================================================================================================


@NumbaChecker("uint32[:](int32, int32, int32)")
def get_reflection_n_estimation(p: int, q: int, n: int) -> np.array:
    """
    Estimates the number of tiles the tiling will have.
    :param p: int = number of edges
    :param q: int = number of polys per vertex
    :param n: int = number of layers (reflective)
    :return: np.array[np.uint32] = number of tildes per layer
    """
    lengths = np.empty((n,), dtype=np.uint32)
    lengths[0] = 0
    lengths[1] = p

    if q == 3:
        k = (p - 2) - 2
        for i in range(2, n):
            lengths[i] = k * lengths[i - 1] - lengths[i - 2]
        lengths[0] = 1
        return lengths

    elif q % 2 == 1:
        k = p - 1
        delta = int((q - 1) // 2)
        # fillers[:, 0] == 1st order; fillers[:, 1] == 2nd order
        fillers = np.zeros((n, 2), dtype=np.uint32)
        if delta < n:
            fillers[delta, 1] = 2 * p

        for i in range(2, n):
            if i <= delta:
                lengths[i] = k * lengths[i - 1]
            else:
                i_delta = i - delta
                i_1 = i - 1
                # update filler of 1st order
                fillers[i, 0] = int(fillers[i_delta, 1] / 2)
                # update filler of 2nd order
                fillers[i, 1] = 2 * ((p - 2) * (lengths[i_delta] - fillers[i_delta, 0] - fillers[i_delta, 1]) + \
                                     (p - 3) * (fillers[i_delta, 0] + fillers[i_delta, 1]))
                # update lengths
                lengths[i] = k * lengths[i_1] - fillers[i_1, 1] - fillers[i, 0] - fillers[i_1, 0]

        lengths[0] = 1
        return lengths

    else:  # q % 2 == 0
        k = p - 1
        delta = int(q // 2)
        fillers = np.zeros((n,), dtype=np.uint32)
        if delta < n:
            fillers[delta] = p

        for i in range(2, n):
            if i <= delta:
                lengths[i] = k * lengths[i - 1] - fillers[i]
            else:
                i_delta = i - delta
                i_1 = i - 1
                fillers[i] = (p - 2) * (lengths[i_delta] - fillers[i_delta]) + (p - 3) * fillers[i_delta]
                lengths[i] = k * lengths[i_1] - fillers[i] - fillers[i_1]
        lengths[0] = 1
        return lengths


@NumbaChecker(["uint8[::1](int64, int64, float64, complex128[:, ::1], uint32[::1], uint8[::1], float64)",
               "uint8[::1](int64, int64, float64, complex128[:, ::1], uint32[::1], uint16[::1], float64)",
               "uint8[::1](int64, int64, float64, complex128[:, ::1], uint32[::1], uint32[::1], float64)"])
def generate(p: int, q: int, r: float, sector_polys: np.array, sector_lengths: np.array,
             edge_array: np.array, mangle: float) -> np.array:
    """
    Generates the tiling with the given parameters p, q, n.
    Time-complexity: O(p^2 m(p, q, n) + n), with m(p, q, n) is the number of polygons
    :param p: int = number of edges
    :param q: int = number of polys per vertex
    :param r: float = radius of the fundamental polygon
    :param sector_polys: np.array[complex][p + 1, x] = array containing the polygons [[center, vertices],...]
    :param sector_lengths: np.array[int] = length
    :param edge_array: np.array[int] = binary of number represents which edges are free
    (will be determined, just give it an array with edge_array.shape[0] == sector_polys.shape[0])
    :param mangle: float = rotation of the center polygon
    :return: np.array[np.uint8] = stores for every polygon which reflection level it has
    """
    dphi = PI2 / p
    phis = np.array([dphi * i + mangle for i in range(p)])  # p

    # most inner polygon
    sector_polys[0, 0] = 0
    sector_polys[0, 1:] = r * np.exp(1j * phis)  # p

    # prepare reflection array
    reflection_levels = np.empty(sector_polys.shape[0], dtype=np.uint8)
    reflection_levels[0] = 0

    # prepare edge_array
    edges = int(2 ** p - 1)
    # eliminate parent edge
    edges ^= 1 << (p - 1)
    edge_array.fill(edges)  # m/p
    # for first poly create only one neighbor
    edge_array[0] = 1

    c = 1
    counter_shift = 0
    for layer_index, layer_size in enumerate(sector_lengths[:-1]):
        next_level_counter = 0

        # check for filler on sector boundary (I think it should detect both)
        connection = any_close_matrix(sector_polys[c - 1] * np.exp(- 1j * dphi), sector_polys[counter_shift])
        if connection.shape[0] == 2 and c > 3:
            # block first child as it would resemble last poly in parents layer
            edge_array[counter_shift] ^= 1 << (connection[0, 0] - 1)

        for j in range(counter_shift, layer_size + counter_shift):
            poly = sector_polys[j]
            r = np.abs(poly[0])

            if j > 1:
                # check if parent poly shares edge with last created child -> filler of 1st order
                connection = any_close_matrix(sector_polys[c - 1], poly)  # (p+1)^2
                if connection.shape[0] == 2 and c > 3:
                    # block edges in number-bit-array (see. GRK.__init__ for explanation)
                    edge_array[c - 1] ^= 1 << (connection[1, 1] - 1)
                    edge_array[j] ^= 1 << (connection[0, 0] - 1)

            for i, vertex in enumerate(poly[1:]):  # p loop execs
                """
                Algorithm:
                 1. shift vertex into origin
                 2. rotate poly such that two vertices are on the x-axis
                 3. reflection on the x-axis (inversion of the imaginary part)
                 4. rotate poly back to original orientation (it is now reflected)
                 5. shift poly back to original position
                """
                if not (edge_array[j] & 1 << i):  # not important (time complexity)
                    continue

                z = poly.copy()  # p  + 1
                array_trans.morigin(p, vertex, z)  # p + 1
                phi = np.angle(z[1:][(i + 1) % p])  # 1
                array_trans.mrotate(p, phi, z)  # p + 1
                z = np.conjugate(z)  # p + 1
                array_trans.mrotate(p, - phi, z)  # p + 1
                array_trans.morigin(p, - vertex, z)  # p + 1

                if r <= np.abs(z[0]):
                    sector_polys[c, 0] = z[0]
                    sector_polys[c, 1:] = np.roll(np.flip(z[1:]), i + 1)  # p

                    # save level of polygons
                    reflection_levels[c] = reflection_levels[j] + 1  # 1

                    if i == 0:
                        # shares edge with former polygon -> filler of 2nd Order
                        connection = any_close_matrix(sector_polys[c], sector_polys[c - 1])  # (p+1)^2
                        if connection.shape[0] == 2 and c > 2:
                            # block edges in number-bit-array (see. GRK __init__ for explanation)
                            edge_array[c] ^= 1 << (connection[1, 1] - 1)
                            edge_array[c - 1] ^= 1 << (connection[0, 0] - 1)

                    elif q == 3:
                        # close first edge because of sibling
                        edge_array[c] ^= 1
                        # close last edge because of sibling
                        if not (edge_array[c - 1] & 1 << (p - 2)):
                            # if filler polygon of first order the second to last edge will be closed
                            # 1 << (p - 2) checks for second to last edge
                            # in this case, the sibling will be on the third to last edge (1 << (p - 3))
                            edge_array[c - 1] ^= 1 << (p - 3)
                        else:
                            # if polygon is a regular polygon, the sibling will be on the second to last edge (1 << (p - 2))
                            edge_array[c - 1] ^= 1 << (p - 2)

                    """
                    Theoretically possible to shift before neighbor comparison.
                    However, even if this would avoid some (maybe useless) calculations it can be important if the
                    graph should be expanded later on.
                    """
                    c += 1
                    next_level_counter += 1
                    if next_level_counter == sector_lengths[layer_index + 1]:
                        break

            if next_level_counter == sector_lengths[layer_index + 1]:
                break

        counter_shift += layer_size

    return reflection_levels

# Methods ==============================================================================================================
