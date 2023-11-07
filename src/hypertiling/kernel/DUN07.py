import numpy as np
import copy
import math
from scipy.stats import circmean
from hypertiling.ion import htprint
from hypertiling.util import fund_radius
from hypertiling.representations import w2p_xyt, p2w_xyt_vector, w2p_xyt_vector
from hypertiling.kernel_abc import Tiling
from hypertiling.kernel.SR_util import HyperPolygon


class Dunham(Tiling):
    """
    A more or less literal, unoptimized, implementation of the tiling algorithm by Douglas Dunham,
    translated to Python; specifically, this is the improved version published in [Dun07]

    Note that this kernel internally uses Weierstrass (hyperboloid) arithmetic. 

    Sources:
    - [Dun86] Dunham, Douglas. "Hyperbolic symmetry." Symmetry. Pergamon, 1986. 139-153.
    - [Dun07] Dunham, Douglas. "An algorithm to generate repeating hyperbolic patterns." the Proceedings of ISAMA (2007): 111-118.
    - [Dun09] Dunham, Douglas. "Repeating Hyperbolic Pattern Algorithms — Special Cases." unpublished (2009)
    - [JvR12] von Raumer, Jakob. "Visualisierung hyperbolischer Kachelungen", Bachelor thesis (2012), unpublished

    """

    def __init__(self, p, q, n):
        super().__init__(p, q, n)

        if p == 3 or q == 3:
            raise ValueError("[hypertiling] Error: p=3 or q=3 is currently not supported!")

        # prepare list to store polygons 
        self.polygons = []

        # fundamental polygon of the tiling
        self._create_fundamental_polygon()

        # store transformation which reflect the fundamental polygon across its edges
        self._compute_edge_reflections()

        # prepare some more variables
        self._prepare_exposures()

        # construct tiling
        self._generate()

    def _create_fundamental_polygon(self):
        """
        Constructs the vertices of the fundamental hyperbolic {p,q} polygon

        Parameters
        ----------
        center : str
            decides whether the fundamental cell is construct centered at the origin ("cell", default) 
            or with the origin being one of its vertices ("vertex")
        rotate_by : float
            angle of rotation of the fundamental polygon, default is the magic angle mangle
        """

        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p)

        for i in range(self.p):
            z = complex(math.cos(i * self.phi), math.sin(i * self.phi))  # = exp(i*phi)
            z = z / abs(z)
            z = r * z
            polygon.verticesP[i] = z
        # transform to Weierstrass coordinates
        # TODO: Use hyperboloid coordinates already during construction
        self.fund_poly = p2w_xyt_vector(polygon.verticesP)

    # ---------- the interface --------------

    def __iter__(self):
        for poly in self.polygons:
            # (center, vertex_1, vertex_2, ..., vertex_p)
            yield np.roll(w2p_xyt_vector(poly),1)
            

    def __len__(self):
        return len(self.polygons)

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

    def _prepare_exposures(self):
        """
        We define the exposure of a p-gon in terms of the number of edges 
        it has in common with the next layer.
        A p-gon has minimum exposure if it has the fewest edges in common with 
        the next layer, and thus shares an edge with the previous layer.
        A p-gon has maximum exposure if it has the most edges in common with the 
        next layer, and thus only shares a vertex with the previous layer.
        We abbreviate these values as min_exp and max_exp, respectively
        """

        self.max_exp = self.p - 2
        self.min_exp = self.p - 3

    def _compute_edge_reflections(self):
        """
        Dunham's algorithm [Dun07] requires a list of reflection transformation
        across the edges of a fundamental polygon. Those are computed here
        """

        self.edge_tran = []

        # reflection transformation from [Dun86]
        tb = 2 * np.arccosh(np.cos(np.pi / self.q) / np.sin(np.pi / self.p))
        reflecty = np.array([[-np.cosh(tb), 0, np.sinh(tb)], [0, 1, 0], [-np.sinh(tb), 0, np.cosh(tb)]])

        # iterate over edges
        for i in range(self.p):
            j = int((i + 1) % self.p)

            # compute angle of midpoint of edge
            phi1 = np.arctan2(self.fund_poly[i][1], self.fund_poly[i][0])
            phi2 = np.arctan2(self.fund_poly[j][1], self.fund_poly[j][0])
            phi = circmean([phi1, phi2])

            # compute associated rotation matrix
            rotphi = rotationW(phi)
            rotinv = rotationW(-phi)

            # 1. rotate such that edge becomes parallel to y-axis
            # 2. perform reflection in x-direction on radius of fundamental cell
            # 3. rotate back
            edge_trafo = rotinv @ reflecty @ rotphi

            # wrap as class object
            # note: the proper usage of the orientation value is unexplained in the Dunham's papers, we stick 
            # with -1 in combination with (0,1,2,3,...) as edge indices, since this produces a proper tiling
            # compare [JvR12] section 3.2 for further details
            self.edge_tran.append(DunhamTransformation(edge_trafo, -1, i))

    def _draw_pgon_pattern(self, trans):
        """
        Apply transformation to copy of fundamental polygon and add resulting polygon to tiling
        Since hypertiling uses Poincare disk coordinates, but this kernel uses Weierstrass (hyperboloid)
        coordinates, this requires some transformations between the two representations
        """
        vrtsW = copy.deepcopy(self.fund_poly)
        vrtsW = [trans.matrix @ k for k in vrtsW]
        self.polygons.append(vrtsW)

    # increment transformation
    def _add_to_tran(self, tran, shift):
        if shift % self.p == 0:
            return tran
        else:
            return self._compute_tran(tran, shift)

    # helper
    def _compute_tran(self, tran, shift):
        newEdge = (tran.p_position + tran.orientation * shift) % self.p

        res = tran * self.edge_tran[newEdge]
        return res

    def _replicate_motif(self, poly, initialTran, layer, exposure):
        """
        central recursion step
        """

        # Draw polygon
        self._draw_pgon_pattern(initialTran)

        # Proceed to desired depth
        if layer < self.n:
            # Determine which vertex to start at
            min_exposure = (exposure == self.min_exp)
            pShift = 1 if min_exposure else 0
            verticesToDo = self.p - 3 if min_exposure else self.p - 2

            # Iterate over vertices
            for i in range(1, verticesToDo + 1):
                first_i = (i == 1)
                pTran = self._compute_tran(initialTran, pShift)
                qSkip = -1 if first_i else 0
                qTran = self._add_to_tran(pTran, qSkip)
                pgonsToDo = self.q - 3 if first_i else self.q - 2

                # Iterate about a vertex
                for j in range(1, pgonsToDo + 1):
                    first_j = (j == 1)
                    newExposure = self.min_exp if first_j else self.max_exp
                    self._replicate_motif(poly, qTran, layer + 1, newExposure)
                    qTran = self._add_to_tran(qTran, -1)

                # Advance to next vertex
                pShift = (pShift + 1) % self.p

    def _replicate(self, poly):
        """
        Top-level driver routine;
        draws the second layer and kicks off the recursion
        """

        # Add fundamental polygon to list
        identity = DunhamTransformation(np.eye(3), -1, 0)
        self._draw_pgon_pattern(identity)

        if self.n == 1:
            return

        # Iterate over each vertex
        for i in range(1, self.p + 1):
            qTran = self.edge_tran[i - 1]

            # Iterate about a vertex
            for j in range(1, self.q - 2 + 1):
                exposure = self.min_exp if (j == 1) else self.max_exp
                self._replicate_motif(poly, qTran, 2, exposure)
                qTran = self._add_to_tran(qTran, -1)

    def _generate(self):
        self._replicate(self.fund_poly)


class DunhamTransformation:
    """
    Transformations contain:
    - the transformation matrix
    - the orientation (-1 or +1)
    - an index of the edge across which the last transformation was made
    """

    def __init__(self, matrix, orientation, p_position):
        self.matrix = matrix
        self.orientation = orientation
        self.p_position = p_position

    def __mul__(self, other):
        # specify how trafos are multiplied
        new_matrix = self.matrix @ other.matrix
        new_orient = self.orientation * other.orientation
        new_p_pos = other.p_position
        return DunhamTransformation(new_matrix, new_orient, new_p_pos)


def rotationW(phi):
    # return Weierstrass rotation matrix
    return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])


if __name__ == "__main__":
    from hypertiling.graphics.plot import plot_tiling
    import matplotlib.pyplot as plt
    import time

    t1 = time.time()
    t = Dunham(5, 4, 3)
    print(f"Took: {time.time() - t1} s")
    plot_tiling(t, np.ones(len(t)), alpha=0.5, ec="k")
    plt.show()