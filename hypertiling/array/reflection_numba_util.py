from typing import Tuple, Callable
from numba import njit
import numpy as np

# Variables ============================================================================================================

PI2 = 2 * np.pi


# Variables ============================================================================================================
# Transformations ======================================================================================================

@njit()
def moeb_origin_trafo(z: np.array, z0: np.complex128) -> np.array:
    """
    Shifts the origin of the points in z such that z0 -> 0. Leaves boundary |z| = 1 circle invariant
    :param z: np.array[complex] = array of points to shift
    :param z0: complex = new origin in the disc
    :result: np.array[complex] = array of shifted points
    """
    num = z - z0
    denom = 1 - z * np.conjugate(z0)
    return num / denom


@njit()
def moeb_rotate_trafo(z: np.array, phi: float) -> np.array:
    """
    Rotate the points described in z around the angle phi
    :param z: np.array[complex] = array of points to rotate
    :param phi: float = angle for the rotation
    :result: np.array[complex] = array of the rotated points
    """
    return z * np.exp(complex(0, phi))


# Transformations ======================================================================================================
# Assistance ===========================================================================================================

@njit()
def any_close_matrix(zs1: np.array, zs2: np.array, tol: float = 1e-12):
    return np.argwhere(np.abs(zs1 - zs2.reshape(zs2.shape[0], 1)) <= tol)


@njit()
def generate_raw(poly: np.array) -> np.array:
    """
    Generates the neigboring polygons for a single polygon poly
    :param poly: np.array[np.complex128][p + 1] = polygon to grow
    :return: np.array[np.complex128][p] = centers of the neigboring polygons
    """
    reflection_centers = np.empty((poly.shape[0] - 1,), dtype=np.complex128)
    for k, vertex in enumerate(poly[1:]):
        z = moeb_origin_trafo(poly, vertex)
        phi = np.angle(z[1:][(k + 1) % (poly.shape[0] - 1)])
        z = moeb_rotate_trafo(z[0], - phi)
        z = np.conjugate(z)
        z = moeb_rotate_trafo(z, phi)
        z = moeb_origin_trafo(z, - vertex)
        reflection_centers[k] = z
    return reflection_centers


# Assistance ===========================================================================================================
# Methods ==============================================================================================================

@njit()
def get_ns(geo_atts: Tuple[int, int, int]) -> np.array:
    """
    Calculates the number of tildes the tiling will have.
    :param geo_atts: Tuple[int, int, int] = (p, q, n)
    :return: np.array[np.uint32] = number of tildes per layer
    """
    lengths = np.empty((geo_atts[2],), dtype=np.uint32)
    lengths[0] = 0
    lengths[1] = (geo_atts[1] - 2) * geo_atts[0]
    fac = (geo_atts[1] - 2) * (geo_atts[0] - 2) - 2
    for i in range(2, geo_atts[2]):
        lengths[i] = fac * lengths[i - 1] - lengths[i - 2]

    lengths[0] = 1
    return lengths


@njit()
def generate(geo_atts: Tuple[int, int, int], r: float, sector_polys: np.array, sector_lengths: np.array, degtol: float):
    """
    Generates the tiling of the polygon
    :param geo_atts: Tuple[int, int, int] = [p, q, n]
    :param r: float = radius of the fundamental polygon
    :param sector_polys: np.array[complex][p + 1, x] = array containing the polygons [[center, vertices],...]
    :param sector_lengths: np.array[int] = length
    :param roll_f: callable = numba compiled callable for the correct ordering of the vertices in sector_polys
    :param degtol: float = tolerance at the boundary
    :return: void
    """
    dphi = PI2 / geo_atts[0]
    phis = np.array([dphi * i for i in range(geo_atts[0])])

    # most inner polygon
    sector_polys[0, 0] = 0
    sector_polys[0, 1:] = r * np.exp(1j * phis)

    c = 1
    stop = np.sum(sector_lengths)

    # prepare edge_array

    edge_array = np.empty((stop,), dtype=np.uint8)
    edges = int(2 ** geo_atts[0] - 1)

    # eliminate parent edge
    edges ^= 1 << (geo_atts[0] - 1)

    edge_array.fill(edges)
    # for first poly create only one neighbor
    edge_array[0] = 1

    boundary = PI2 / geo_atts[0] + (degtol / 360 * PI2)
    for j, poly in enumerate(sector_polys[:-1]):
        for i, vertex in enumerate(poly[1:]):
            """
            Algorithm:
             1. shift vertex into origin
             2. rotate poly such that two vertices are on the x-axis
             3. reflection on the x-axis (inversion of the imaginary part)
             4. rotate poly back to original orientation (it is now reflected)
             5. shift poly back to original position
            """
            if not (edge_array[j] & 1 << i):
                continue

            z = moeb_origin_trafo(poly, vertex)
            phi = np.angle(z[1:][(i + 1) % geo_atts[0]])
            z = moeb_rotate_trafo(z, - phi)
            z = np.conjugate(z)
            z = moeb_rotate_trafo(z, phi)
            z = moeb_origin_trafo(z, - vertex)

            angle = np.angle(z[0])
            if angle > boundary:
                break

            if angle >= 0:
                sector_polys[c, 0] = z[0]
                sector_polys[c, 1:] = np.roll(np.flip(z[1:]), i + 1)

                # shares edge with former polygon (sibling)
                connection = any_close_matrix(sector_polys[c], sector_polys[c - 1])
                if connection.shape[0] == 2 and c > 2:
                    edge_array[c] ^= 1 << (connection[1, 1] - 1)
                    edge_array[c - 1] ^= 1 << (connection[0, 0] - 1)

                # check if poly shares edge with next parent (parents sibling) #filler
                connection = any_close_matrix(sector_polys[c], sector_polys[j + 1])
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
                    return 0
# Methods ==============================================================================================================
