import numpy as np
import copy
from ..ion import htprint
from ..representations import w2p_xyt, p2w_xyt
from .SR_base import KernelStaticBase, HyperPolygon


class LegacyDunham(KernelStaticBase):
    """
    This kernel implements the "original" construction algorithm by D. Dunham, 
    published in "Hyperbolic symmetry." Symmetry. Pergamon, 1986. 139-153
    from here on references as [Dun86].

    Note that this algorithm does not work with the parameters stated in [Dun86], as
    some cells are missing. We adjusted the parameters to generate a full tiling,
    however at the cost of producing lots of duplicates!

    This kernel is only implemented for legacy reasons. There is a more recent version of 
    Dunham's algorithm implemented in the DUN kernel.

    The algorithm uses Weierstraß (hyperboloid) coordinates; since those are not natively supported
    by our HyperPolygon class we need transformation functions provided in DUN_util.py
    """

    def __init__ (self, p, q, n, **kwargs):
        super(LegacyDunham, self).__init__(p, q, n, **kwargs)

        htprint("Warning", "This kernel is only implemented for legacy reasons and may not be fully functional! It is advised to use a different kernel!")

        # reflection and rotation matrices
        self.b = np.arccosh(np.cos(np.pi / q) / np.sin(np.pi / p))

        self.ReflectPgonEdge = np.array([[-np.cosh(2 * self.b), 0, np.sinh(2 * self.b)],
                                         [0, 1, 0],
                                         [-np.sinh(2 * self.b), 0, np.cosh(2 * self.b)]])
        self.ReflectEdgeBisector = np.array([[1, 0, 0],
                                             [0, -1, 0],
                                             [0, 0, 1]])
        self.ReflectHypotenuse = np.array([[np.cos(2 * np.pi / p), np.sin(2 * np.pi / p), 0],
                                           [np.sin(2 * np.pi / p), -np.cos(2 * np.pi / p), 0],
                                           [0, 0, 1]])

        self.RotP  = self.ReflectHypotenuse @ self.ReflectEdgeBisector
        self.RotQ  = self.ReflectPgonEdge @ self.ReflectHypotenuse
        self.Rot2P = self.RotP @ self.RotP
        self.Rot3P = self.Rot2P @ self.RotP
        self.RotCenterG = np.eye(3)     # will be manipulated in self.generate()
        self.RotCenterR = np.eye(3)     # will be manipulated in self.replicate()

        # fundamental polygon of the tiling
        self._create_first_layer(self.phi/2)

        # construct tiling
        self._generate()


    def _generate(self):
        if self.n == 1:
            return

        for _ in range(1, self.p+1):
            RotVertex = self.RotCenterG @ self.RotQ
            self._replicate(RotVertex, self.n - 2, "Edge")
            for _ in range(1, self.q - 3 + 1):
                RotVertex = RotVertex @ self.RotQ
                self._replicate(RotVertex, self.n - 2, "Vertex")

            self.RotCenterG = self.RotCenterG @ self.RotP

    
    def _replicate(self, InitialTran, LayersToDo, AdjacencyType):
        
        # create deep copy, apply current transform and add to list
        poly = copy.deepcopy(self.fund_poly)
        transformW_poly(poly, InitialTran)
        self.polygons.append(poly)

        ExposedEdges = 0
        VertexPgons = 0

        # iterate layers
        if LayersToDo > 0:
            if AdjacencyType == "Edge":
                ExposedEdges = self.p - 3
                self.RotCenterR = InitialTran @ self.Rot3P
            if AdjacencyType == "Vertex":
                ExposedEdges = self.p - 2
                self.RotCenterR = InitialTran @ self.Rot2P

            # iterate exposed edges
            for j in range(1, ExposedEdges + 1):
                RotVertex = self.RotCenterR @ self.RotQ
                self._replicate(RotVertex, LayersToDo - 1, "Edge")
                if j < ExposedEdges:
                    VertexPgons = self.q - 1  # -3 in [Dun86]
                elif j == ExposedEdges:
                    VertexPgons = self.q - 2  # -4 in [Dun86]

                # iterate rotations about that vertex
                for _ in range(1, VertexPgons + 1):
                    RotVertex = RotVertex @ self.RotQ
                    self._replicate(RotVertex, LayersToDo - 1, "Vertex")

                # increment transformation
                self.RotCenterR = self.RotCenterR @ self.RotP


    def _draw_pgon_pattern(self, Transformation):
        # create permanent copy of fundamental polygon
        poly = copy.deepcopy(self.fund_poly)
        # apply transformation
        transformW_poly(poly, Transformation)
        # draw, i.e. add to list
        self.polygons.append(poly)



    def add_layer(self):
        htprint("Warning", "The requested function is not implemented! Please use a different kernel!")
        return
    


def transformW_poly(polygon: HyperPolygon, transformation):
    """
    Apply Weierstraß transformation matrix to entire HyperPolygon, i.e. vertices and center coordiantes
    """
    new_verts = np.zeros_like(polygon.verticesP)
    for i, pointP in enumerate(polygon.verticesP):
        new_verts[i] = transformW_site(pointP, transformation)
    polygon.verticesP = new_verts


def transformW_site(pointP: np.complex128, transformation):
    """
    Apply Weierstraß transformation to Poincare site
    1. Transform site from Poincare to Weierstraß
    2. Apply Weierstraß transformation
    3. Transform back
    """
    return w2p_xyt(transformation @ p2w_xyt(pointP))