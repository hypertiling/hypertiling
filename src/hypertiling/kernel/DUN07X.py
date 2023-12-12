from typing import Tuple
import numpy as np
from scipy.stats import circmean
from hypertiling.util import fund_radius
from hypertiling.kernel_abc import Tiling
from hypertiling.util import n_cell_centered
from hypertiling.representations import w2p_xyt, w2p_xyt_vector, p2w_xyt
from hypertiling.ion import htprint
from hypertiling.check_numba import NumbaChecker, DelayChecker


class DunhamX(Tiling):
    """
    Performance optimized implementation of the tiling algorithm by Douglas Dunham, published in [Dun07]
    It achieves a factor 10-15x speed up compared to the literal, unoptimized implementation,
    which is available as kernel "Dunham" in DUN07.py

    Note that this kernel internally uses Weierstrass (hyperboloid) arithmetic. 

    Sources:
    - [Dun86] Dunham, Douglas. "Hyperbolic symmetry." Symmetry. Pergamon, 1986. 139-153.
    - [Dun07] Dunham, Douglas. "An algorithm to generate repeating hyperbolic patterns." the Proceedings of ISAMA (2007): 111-118.
    - [Dun09] Dunham, Douglas. "Repeating Hyperbolic Pattern Algorithms — Special Cases." unpublished (2009)
    - [JvR12] von Raumer, Jakob. "Visualisierung hyperbolischer Kachelungen", Bachelor thesis (2012), unpublished

    """

    def __init__(self, p: int, q: int, n: int):
        """
        Initialize a tiling with parameters p, q, n
        :param p: int = number of edges per polygon
        :param q: int = number of polygons meeting at a vertex
        :param n: int = number of layers
        """
        super().__init__(p, q, n)

        if p == 3 or q == 3:
            raise ValueError("[hypertiling] Error: p=3 or q=3 is currently not supported!")

        # prepare list to store polygons
        self.length = n_cell_centered(p, q, n)
        self.polygons = np.empty((self.length, p + 1, 3), dtype=np.float64)

        # fundamental polygon of the tiling
        self.polygons[0] = self._create_fundamental_polygon()

        # store transformation which reflect the fundamental polygon across its edges
        self.edge_trans, self.trans_props = self._compute_edge_reflections()
        # self.edge_trans = [trans, ...]
        # self.trans_props = [[orient, pos], ...]

        # prepare some more variables
        self.max_exp = self.p - 2
        self.min_exp = self.p - 3

        # construct tiling
        self._generate()

    def _create_fundamental_polygon(self) -> np.array:
        """
        Calculates the fundamental polygon
        :return: np.array[p + 1, 3] = [center, vertex1, ...] in Weierstrass coordinates
        """
        zs = np.empty((self.p + 1,), dtype=np.complex128)
        phis = np.arange(0, self.p) * self.phi
        zs_ = np.cos(phis) + 1j * np.sin(phis)
        zs[:self.p] = (zs_ / np.abs(zs_)) * fund_radius(self.p, self.q)
        zs[self.p] = 0
        return p2w_xyt_vector(zs)

    def _compute_edge_reflections(self) -> Tuple[np.array, np.array]:
        """
        Calculate the fundamental transformations on the edges
        :return: Tuple[np.array[p, 3, 3], np.array[p, 2]] = ([trans1, ...], [[orient, pos], ...]]) transformations and
                                                            their properties
        """
        # reflection transformation from [Dun86]
        tb = 2 * np.arccosh(np.cos(np.pi / self.q) / np.sin(np.pi / self.p))
        reflecty = np.array([[-np.cosh(tb), 0, np.sinh(tb)], [0, 1, 0], [-np.sinh(tb), 0, np.cosh(tb)]])

        # iterate over edges
        trans = np.empty((self.p, 3, 3), dtype=np.float64)
        trans_props = np.empty((self.p, 2), dtype=np.int32)
        for i in range(self.p):
            j = (i + 1) % self.p
            phi = circmean([np.arctan2(self.polygons[0, i, 1], self.polygons[0, i, 0]),
                            np.arctan2(self.polygons[0, j, 1], self.polygons[0, j, 0])])
            trans[i] = rotationW(-phi) @ reflecty @ rotationW(phi)
            trans_props[i, 0] = -1
            trans_props[i, 1] = i
        return trans, trans_props

    # ---------- the interface --------------
    def __iter__(self) -> np.array:
        """
        Iterate over the polygons in the tiling
        :yield: np.array[p + 1] = [center, vertex1, ...] polygon in Poincaré coordinates
        """
        for poly in self.polygons:
            # (center, vertex_1, vertex_2, ..., vertex_p)
            yield np.roll(w2p_xyt_vector(poly), 1)

    def __len__(self) -> int:
        """
        Returns the number of polygons in the tiling
        :return: int = number of polygons
        """
        return self.length

    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index in Poincare disk coordinates
        Since this kernel's internal arithmetic is done in Weierstrass representation,
        this requires some coordinate transform
        Time-complexity: O(1)
        Overwrites method of base class
        :param index: int = index of the polygon
        :return: np.array[np.complex128][p] = vertices of the polygon
        """
        return w2p_xyt_vector(self.polygons[index][:-1])

    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index in Poincare disk coordinates
        Since this kernel's internal arithmetic is done in Weierstrass representation,
        this requires a coordinate transform
        Time-complexity: O(1)
        Overwrites method of base class
        :param index: int = index of the polygon
        :return:  -> np.complex128: = center of the polygon
        """
        return w2p_xyt(self.polygons[index][-1])

    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: float = angle of the polygon
        """
        return np.angle(self.get_center(index))

    def get_layer(self, index: int) -> int:
        htprint("Warning", "Layer information is currently not implemented in this kernel, doing nothing ...")

    def get_sector(self, index: int) -> int:
        htprint("Warning", "No sectors used in this kernel, doing nothing ...")

    def add_layer(self):
        htprint("Warning", "The requested function is not implemented! Please use a different kernel!")

    # ---------- the algorithm --------------

    def _generate(self):
        """
        Generate the tiling according to the Dunham algorithm
        :return: void
        """
        if self.n == 1:
            return

        # Iterate over each vertex
        c = 1
        for i in range(1, self.p + 1):
            trans = np.copy(self.edge_trans[i - 1])
            trans_orient = self.trans_props[i - 1, 0]
            trans_pos = self.trans_props[i - 1, 1]

            # Iterate about a vertex
            for j in range(1, self.q - 2 + 1):
                exposure = self.min_exp if (j == 1) else self.max_exp
                c = generate_dun(self.p, self.q, self.n, self.polygons, c, self.edge_trans, self.trans_props, trans,
                                 trans_orient, trans_pos, 2, exposure, self.min_exp, self.max_exp)
                trans, trans_orient, trans_pos = add_trans(self.p, self.trans_props, self.edge_trans, trans,
                                                           trans_orient, trans_pos, -1)


# ---------- numba optimized functions --------------

@NumbaChecker(
    "Tuple((float64[:,::1], int32, int32))(int32, int32[:,::1], float64[:,:,::1], float64[:,::1], int32, int32, int32)")
def comp_tran(p: int, transs_props: np.array, transs: np.array, trans: np.array, trans_orient: int, trans_pos: int,
              shift: int) -> Tuple[np.array, int, int]:
    """
    Calculate the new transformation for the edge
    :param p: int = number of edges
    :param transs_props: np.array[p, 2] = [[orient, pos], ...] for edge trans
    :param transs: np.array[p, 3, 3] = [trans, ...] for edge trans
    :param trans: np.array[3, 3] = transformation (current)
    :param trans_orient: int = orientation of transformation (current) in {-1, 1}
    :param trans_pos: int = position of transformation (current)
    :param shift: int = -1
    :return: Tuple[np.array[3, 3], int, int] = transformation, orientation, position
    """
    new_edge = (trans_pos + trans_orient * shift) % p
    trans = trans @ transs[new_edge]
    trans_orient = trans_orient * transs_props[new_edge, 0]
    trans_pos = transs_props[new_edge, 1]
    return trans, trans_orient, trans_pos


@NumbaChecker(
    "Tuple((float64[:,::1], int32, int32))(int32, int32[:,::1], float64[:,:,::1], float64[:,::1], int32, int32, int32)")
def add_trans(p: int, transs_props: np.array, transs: np.array, trans: np.array, trans_orient: int, trans_pos: int,
              shift: int) -> Tuple[np.array, int, int]:
    """
    Calculate the new transformation for the edge by adding a shift
    :param p: int = number of edges
    :param transs_props: np.array[p, 2] = [[orient, pos], ...] for edge trans
    :param transs: np.array[p, 3, 3] = [trans, ...] for edge trans
    :param trans: np.array[3, 3] = transformation (current)
    :param trans_orient: int = orientation of transformation (current) in {-1, 1}
    :param trans_pos: int = position of transformation (current)
    :param shift: int = -1
    :return: Tuple[np.array[3, 3], int, int] = transformation, orientation, position
    """
    if shift % p == 0:
        return trans, trans_orient, trans_pos
    else:
        return comp_tran(p, transs_props, transs, trans, trans_orient, trans_pos, shift)


@NumbaChecker("float64[:,::1](complex128[::1])")
def p2w_xyt_vector(z_array: np.array) -> np.array:
    """
    Calculate the Weierstrass coordinates for the coordinates given in z_array
    :param z_array: np.array[p + 1] = coordinates in Poincaré coordinates
    :return: np.array[p + 1, 3] = coordinates in Weierstrass coordinates
    """
    result = np.empty((len(z_array), 3), dtype=np.float64)
    for index, z in enumerate(z_array):
        result[index] = p2w_xyt(z)
    return result


@NumbaChecker("float64[:,::1](float64)")
def rotationW(phi: float) -> np.array:
    """
    Calculates the rotational matrix for a given phi
    :param phi: float = angle of the rotation
    :return: np.array[3, 3] = rotation matrix
    """
    # return Weierstrass rotation matrix
    c = np.cos(phi)
    s = np.sin(phi)
    rot = np.zeros((3, 3), dtype=np.float64)
    rot[0, 0] = c
    rot[1, 1] = c
    rot[0, 1] = - s
    rot[1, 0] = s
    rot[2, 2] = 1
    return rot


@DelayChecker(
    "int32(int32, int32, int32, float64[:,:,::1], int32, float64[:,:,::1], int32[:,::1], float64[:,::1], int32, int32, int32, int32, int32, int32)")
def generate_dun(p: int, q: int, n: int, polygons: np.array, polygon_counter: int, transs: np.array,
                 transs_props: np.array, trans_init: np.array, trans_init_orient: int, trans_init_pos: int, layer: int,
                 exposure: int, min_exp: int, max_exp: int) -> int:
    """
    Calculates the tiling and stores the polygons in polygons. Returns the number of constructed polygons
    :param p: int = number of edges
    :param q: int = number of polygons at a vertex
    :param n: int = number of layers to construct
    :param polygons: np.array[m, p + 1] = [[center, vertex1, ...], ...] = polygons in the tiling
    :param polygon_counter: int = counter of how many polygons are already constructed
    :param transs: np.array[p, 3, 3] = [trans, ...] for edge trans
    :param transs_props: np.array[p, 2] = [[orient, pos], ...] for edge trans
    :param trans_init: np.array[3, 3] = transformation (current)
    :param trans_init_orient: int = orientation of transformation (current) in {-1, 1}
    :param trans_init_pos: int = position of transformation (current)
    :param layer: int = current layer in construction
    :param exposure: int = number of open edges
    :param min_exp: int = minimal number of open edges
    :param max_exp: int = maximal number of open edges
    :return: int = polygon counter after construction
    """
    # add polygon to polygon array
    polygons[polygon_counter] = polygons[0] @ np.transpose(trans_init)  #
    polygon_counter += 1

    # check for end of recursion
    if layer == n:
        return polygon_counter

    # determine which vertex to start at
    min_exposure = (exposure == min_exp)
    p_shift = 1 if min_exposure else 0
    vertices2do = p - 3 if min_exposure else p - 2

    next_layer = layer + 1
    # iterate over vertices
    for i in range(1, vertices2do + 1):
        first_i = (i == 1)
        q_skip = -1 if first_i else 0
        pgons2do = q - 3 if first_i else q - 2

        p_trans, trans_orient, trans_pos = comp_tran(p, transs_props, transs, trans_init, trans_init_orient,
                                                     trans_init_pos, p_shift)
        q_trans, trans_orient, trans_pos = add_trans(p, transs_props, transs, p_trans, trans_orient, trans_pos, q_skip)

        # iterate about a vertex
        for j in range(1, pgons2do + 1):  # exec q - 2 times  => O((q - 2))
            first_j = (j == 1)
            new_exposure = min_exp if first_j else max_exp
            polygon_counter = generate_dun(p, q, n, polygons, polygon_counter, transs, transs_props, q_trans,
                                           trans_orient, trans_pos, next_layer, new_exposure, min_exp, max_exp)
            q_trans, trans_orient, trans_pos = add_trans(p, transs_props, transs, q_trans, trans_orient, trans_pos, -1)

        # Advance to next vertex
        p_shift = (p_shift + 1) % p

    return polygon_counter


if __name__ == "__main__":
    from hypertiling.graphics.plot import plot_tiling
    import matplotlib.pyplot as plt
    import time

    t1 = time.time()
    t = DunhamX(5, 4, 5)
    print(f"Took: {time.time() - t1} s")
    plot_tiling(t, np.ones(len(t)), alpha=0.5, ec="k")
    plt.show()
