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

@NumbaChecker("uint32[::1](int32, int32, int32)")
def get_ns(p: int, q: int, n: int) -> np.array:
    """
    Calculates the number of tildes the tiling will have.
    Time-complexity: O(n)
    :param p: int = number of edges
    :param q: int = number of polys per vertex
    :param n: int = number of layers (traditional)
    :return: np.array[np.uint32] = number of tildes per layer
    """
    lengths = np.empty((n,), dtype=np.uint32)
    lengths[0] = 0
    lengths[1] = (q - 2) * p
    fac = (q - 2) * (p - 2) - 2
    for i in range(2, n):
        lengths[i] = fac * lengths[i - 1] - lengths[i - 2]

    lengths[0] = 1
    return lengths


@NumbaChecker("uint32[::1](int32, int32, int32)")
def get_reflection_n_estimation(p: int, q: int, n: int) -> np.array:
    """
    Estimates the number of tildes the tiling will have.
    Time-complexity: O(n)
    :param p: int = number of edges
    :param q: int = number of polys per vertex
    :param n: int = number of layers (reflective)
    :return: np.array[np.uint32] = number of tildes per layer
    """
    k = p - 3 if q == 3 else p - 2 if q == 4 else p - 1

    lengths = np.empty((n + 1,), dtype=np.uint32)
    lengths[0] = 1
    lengths[1] = p
    for n in range(2, n + 1):
        lengths[n] = lengths[n - 1] * k

    return lengths


@NumbaChecker(["uint8[::1](int64, int64, int64, float64, complex128[:, ::1], uint32[::1], uint8[::1], int64, float64)",
               "uint8[::1](int64, int64, int64, float64, complex128[:, ::1], uint32[::1], uint16[::1], int64, float64)",
               "uint8[::1](int64, int64, int64, float64, complex128[:, ::1], uint32[::1], uint32[::1], int64, float64)"])
def generate(p: int, q: int, n: int, r: float, sector_polys: np.array, sector_lengths: np.array,
             edge_array: np.array, degtol: float, mangle: float) -> np.array:
    """
    Generates the tiling with the given parameters p, q, n.
    Time-complexity: O(p^2 m(p, q, n) + n), with m(p, q, n) is the number of polygons
    :param p: int = number of edges
    :param q: int = number of polys per vertex
    :param n: int = number of layers (reflective)
    :param r: float = radius of the fundamental polygon
    :param sector_polys: np.array[complex][p + 1, x] = array containing the polygons [[center, vertices],...]
    :param sector_lengths: np.array[int] = length
    :param edge_array: np.array[int] = binary of number represents which edges are free
    (will be determined, just give it an array with edge_array.shape[0] == sector_polys.shape[0])
    :param degtol: float = tolerance at the boundary
    :param mangle: float = rotation of the center polygon
    :return: np.array[np.uint8] = stores for every polygon which reflection level it has
    """
    dphi = PI2 / p
    phis = np.array([dphi * i + mangle for i in range(p)])  # p

    # most inner polygon
    sector_polys[0, 0] = 0
    sector_polys[0, 1:] = r * np.exp(1j * phis)  # p

    c = 1
    stop = np.sum(sector_lengths)  # n

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

    boundary = PI2 / p + (degtol / 360 * PI2)
    for j, poly in enumerate(sector_polys[:-1]):  # m/p loop executions
        if reflection_levels[j] == n:
            # all reflection layers are constructed
            return reflection_levels[:c]
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

            angle = np.angle(z[0])  # 1
            if angle > boundary:
                break

            if angle >= 0:
                sector_polys[c, 0] = z[0]
                sector_polys[c, 1:] = np.roll(np.flip(z[1:]), i + 1)  # p

                # save level of polygons
                reflection_levels[c] = reflection_levels[j] + 1  # 1

                # shares edge with former polygon (sibling)
                # connection = any_close_matrix(sector_polys[c], sector_polys[c - 1])  # (p+1)^2
                connection = any_close_matrix(sector_polys[c], sector_polys[c - 1])  # (p+1)^2
                if connection.shape[0] == 2 and c > 2:
                    edge_array[c] ^= 1 << (connection[1, 1] - 1)
                    edge_array[c - 1] ^= 1 << (connection[0, 0] - 1)

                # check if poly shares edge with next parent (parents sibling) #filler
                # connection = any_close_matrix(sector_polys[c], sector_polys[j + 1])  # (p+1)^2
                connection = any_close_matrix(sector_polys[c], sector_polys[j + 1])  # (p+1)^2
                if connection.shape[0] == 2 and c > 3:
                    edge_array[c] ^= 1 << (connection[1, 1] - 1)
                    edge_array[j + 1] ^= 1 << (connection[0, 0] - 1)

                """
                Theoretically possible to shift before neighbor comparison.
                However, even if this would avoid some (maybe useless) calculations it can be important if the
                graph should be expanded later on.
                """
                c += 1
                if c == stop:
                    return reflection_levels
    return reflection_levels

# Methods ==============================================================================================================
