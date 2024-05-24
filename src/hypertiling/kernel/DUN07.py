import numpy as np
import copy
import math
from scipy.stats import circmean
from hypertiling.ion import htprint
from hypertiling.util import fund_radius
from hypertiling.representations import w2p_xyt, p2w_xyt_vector, w2p_xyt_vector
from hypertiling.kernel_abc import Tiling
from hypertiling.kernel.hyperpolygon import HyperPolygon


class Dunham(Tiling):
    """
    A more or less literal, unoptimized, implementation of the tiling algorithm by Douglas Dunham,
    translated to Python; specifically, this is the improved version published in [Dun07]

    Note that this kernel internally uses Weierstrass (hyperboloid) arithmetic. 
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
        This function constructs the vertices of the fundamental hyperbolic {p,q} polygon.
        """

        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p)

        verts = []
        for i in range(self.p):
            z = complex(math.cos(i * self.phi), math.sin(i * self.phi))  # = exp(i*phi)
            z = z / abs(z)
            z = r * z
            verts.append(z)
        polygon.set_vertices(verts)

        # transform to Weierstrass coordinates (TODO: Use those coordinates already during construction)
        self.fund_poly = p2w_xyt_vector(polygon._vertices)


    # ---------- the interface --------------

    def __iter__(self):
        for poly in self.polygons:
            # (center, vertex_1, vertex_2, ..., vertex_p)
            yield np.roll(w2p_xyt_vector(poly),1)
            

    def __len__(self):
        return len(self.polygons)
    

    def __getitem__(self, idx):
        # (center, vertex_1, vertex_2, ..., vertex_p)
        return np.roll(self.polygons[idx], 1)
    

    def get_polygon(self, index: int) -> HyperPolygon:
        """
        Returns the polygon at index as HyperPolygon object. Method exists mainly for compatibility reasons. Usage is discouraged!

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        HyperPolygon
            Polygon at index.
        """
        htprint("Warning", "Method exists mainly for compatibility reasons. Usage is discouraged!")

        polygon = HyperPolygon(self.p)
        polygon.idx = index
        polygon.layer = None
        polygon.sector = None
        polygon.angle = self.get_angle(index)
        polygon.orientation = None
        polygon.set_polygon = self[index]

        return polygon
    
    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index in Poincare disk coordinates.
        Since this kernel's internal arithmetic is done in Weierstrass representation,
        this requires some coordinate transform.
        
        This method overwrites the method of the base class.

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        np.array
            p vertices of the polygon.

        Notes
        -----
        Time complexity of this method is O(1).
        """
        return w2p_xyt_vector(self.polygons[index][:-1])
    

    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index in Poincare disk coordinates.
        Since this kernel's internal arithmetic is done in Weierstrass representation,
        this requires a coordinate transform.

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        np.complex128
            Center of the polygon.

        Notes
        -----
        Time-complexity: O(1)
        Overwrites method of base class.
        """
        return w2p_xyt(self.polygons[index][-1])


    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.

        Parameters
        ----------
        index : int
            Index of the polygon.

        Returns
        -------
        float
            Angle of the polygon.

        Notes
        -----
        Time-complexity: O(1)
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
        We abbreviate these values as min_exp and max_exp, respectively.
        """

        self.max_exp = self.p - 2
        self.min_exp = self.p - 3

    def _compute_edge_reflections(self):
        """
        Computes a list of reflection transformations across the edges of a fundamental polygon as required by Dunham's algorithm [Dun07].

        Generates a list of DunhamTransformation objects, each representing a reflection transformation associated with an edge of the fundamental polygon.

        Steps:
        1. Computes the reflection transformation matrix using the formula from [Dun86].
        2. Iterates over all edges of the polygon.
        3. Computes the angle of the midpoint of the current edge.
        4. Computes the associated rotation matrices.
        5. Performs the following transformations in sequence:
            - Rotates the edge to make it parallel to the y-axis.
            - Performs a reflection in the x-direction on the radius of the fundamental cell.
            - Rotates the edge back to its original orientation.
        6. Wraps the resulting transformation matrix in a DunhamTransformation object and adds it to the list.

        Note: The proper usage of the orientation value is unexplained in Dunham's papers, 
        so we stick with -1 in combination with (0,1,2,3,...) as edge indices. 
        This produces a proper tiling (compare [JvR12] section 3.2 for further details).
        
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
        This method applies a given transformation to a copy of the fundamental polygon and adds the 
        resulting polygon to the tiling.

        Since hypertiling uses Poincare disk coordinates, but this kernel uses Weierstrass (hyperboloid)
        coordinates, it requires transformations between the two representations.

        Parameters
        ----------
        trans : Transformation
            The transformation to apply to the fundamental polygon.

        Returns
        -------
        None
        """
        vrtsW = copy.deepcopy(self.fund_poly)
        vrtsW = [trans.matrix @ k for k in vrtsW]
        self.polygons.append(vrtsW)


    def _add_to_tran(self, tran, shift):
        """
        Helper function to adjust the transformation used in the Dunham tiling algorithm

        Parameters
        ----------
        tran : object
            The original transformation.
        shift : int
            The shift value used to adjust the transformation.

        Returns
        -------
        object
            The adjusted transformation.
        """
        if shift % self.p == 0:
            return tran
        else:
            return self._compute_tran(tran, shift)


    def _compute_tran(self, tran, shift):
        """
        Helper function to adjust the transformation used in the Dunham tiling algorithm
        
        Parameters
        ----------
        tran : DunhamTransformation
            The original transformation object.
        shift : int
            The amount to shift the transformation.

        Returns
        -------
        DunhamTransformation
            The new transformation object after applying the edge transformations.
        """
        newEdge = (tran.p_position + tran.orientation * shift) % self.p
        res = tran * self.edge_tran[newEdge]
        return res


    def _replicate_motif(self, poly, initialTran, layer, exposure):
        """
        This is the central recursion step for the Dunham tiling algorithm.

        This function draws a polygon pattern, then if the current layer is less than the desired layers `self.n`,
        it determines which vertex to start at based on the exposure. It then iterates over the vertices and polygons,
        updating the transformations and exposures accordingly, and recursively calls itself.

        Parameters
        ----------
        poly : object
            The polygonal object to be replicated.
        initialTran : DunhamTransformation
            The initial transformation to be applied.
        layer : int
            The current layer of the recursion.
        exposure : int
            The current exposure level.

        Notes
        -----
        The function first draws a polygon pattern using `initialTran`. If `layer` is less than `self.n`, it determines
        which vertex to start at based on whether `exposure` is equal to `self.min_exp`. It then iterates over the vertices
        and polygons, updating transformations and exposures, and recursively calls itself.
        """

        # Draw polygon
        self._draw_pgon_pattern(initialTran)

        # Proceed to desired depth
        if layer < self.n:
            # Determine which vertex to start at based on exposure
            min_exposure = (exposure == self.min_exp)
            pShift = 1 if min_exposure else 0
            verticesToDo = self.p - 3 if min_exposure else self.p - 2

            # Iterate over vertices
            for i in range(1, verticesToDo + 1):
                first_i = (i == 1)
                # Compute transformation to be applied
                pTran = self._compute_tran(initialTran, pShift)
                qSkip = -1 if first_i else 0
                # Adjust transformation
                qTran = self._add_to_tran(pTran, qSkip)
                pgonsToDo = self.q - 3 if first_i else self.q - 2

                # Iterate about a vertex
                for j in range(1, pgonsToDo + 1):
                    first_j = (j == 1)
                    # Determine new exposure level
                    newExposure = self.min_exp if first_j else self.max_exp
                    # Recursive call
                    self._replicate_motif(poly, qTran, layer + 1, newExposure)
                    # Adjust transformation
                    qTran = self._add_to_tran(qTran, -1)

                # Advance to next vertex
                pShift = (pShift + 1) % self.p


    def _replicate(self, poly):
        """
        This is the top-level driver routine for the Dunham tiling algorithm.

        This function initiates the drawing of the second layer and starts the recursion.

        Parameters
        ----------
        poly : object
            The polygonal object to be replicated.

        Notes
        -----
        This function first creates an identity transformation and uses it to draw a fundamental polygon pattern.
        If the number of desired layers (self.n) is 1, the function returns immediately.
        Otherwise, it iterates over each vertex of the polygon (self.p), and for each vertex, it further iterates
        over a certain number of polygons (self.q - 2).
        For each polygon, it determines the exposure level, replicates the motif, and modifies the transformation.
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
        """
        This function generates the tiling starting from a fundamental polygon
        """
        self._replicate(self.fund_poly)


class DunhamTransformation:
    """
    The DunhamTransformation class is used to represent transformations that include a transformation matrix,
    an orientation value (-1 or +1), and an index of the last transformation's edge.

    Attributes
    ----------
    matrix : np.array
        The transformation matrix.
    orientation : int
        The orientation of the transformation (-1 or +1).
    p_position : int
        The index of the edge across which the last transformation was made.

    Methods
    -------
    __mul__(self, other)
        Specifies how two `DunhamTransformation` objects are multiplied.
    """

    def __init__(self, matrix, orientation, p_position):
        """
        Parameters
        ----------
        matrix : np.array
            The transformation matrix.
        orientation : int
            The orientation of the transformation (-1 or +1).
        p_position : int
            The index of the edge across which the last transformation was made.
        """

        self.matrix = matrix
        self.orientation = orientation
        self.p_position = p_position

    def __mul__(self, other):
        """
        Specifies how two `DunhamTransformation` objects are multiplied.

        Parameters
        ----------
        other : DunhamTransformation
            The other `DunhamTransformation` object.

        Returns
        -------
        DunhamTransformation
            The product of the multiplication of two `DunhamTransformation` objects.
        """

        new_matrix = self.matrix @ other.matrix
        new_orient = self.orientation * other.orientation
        new_p_pos = other.p_position
        return DunhamTransformation(new_matrix, new_orient, new_p_pos)


def rotationW(phi):
    # return Weierstrass rotation matrix
    return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

