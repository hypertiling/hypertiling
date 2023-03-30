import numpy as np
import copy
from ..ion import htprint
from .SR_base import KernelStaticBase
from .DUN_util import transformW_poly, transformW_site



class LegacyDunham(KernelStaticBase):
    """
    This kernel implements the "original" construction algorithm of D. Dunham (1982)
    The algorithm uses Weierstraß (hyperboloid) coordinates; since those are not natively supported
    by our HyperPolygon class we need transformation functions provided in DUN_util.py
    """

    def __init__ (self, p, q, n, **kwargs):
        super(LegacyDunham, self).__init__(p, q, n, **kwargs)


        if self.center == "vertex":
            htprint("Warning", "Dunham kernel does not support vertex centered tilings yet!")

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

        self._draw_pgon_pattern(InitialTran)

        if LayersToDo > 0:
            if AdjacencyType == "Edge":
                ExposedEdges = self.p - 3
                self.RotCenterR = InitialTran @ self.Rot3P
            if AdjacencyType == "Vertex":
                ExposedEdges = self.p - 2
                self.RotCenterR = InitialTran @ self.Rot2P

            for j in range(1, ExposedEdges + 1):
                RotVertex = self.RotCenterR @ self.RotQ
                self._replicate(RotVertex, LayersToDo - 1, "Edge")
                if j < ExposedEdges:
                    VertexPgons = self.q - 1  # was -3 in Dunhams paper, this seems to be a better value though
                elif j == ExposedEdges:
                    VertexPgons = self.q - 2  # was -4 in Dunhams paper

                for _ in range(1, VertexPgons + 1):
                    RotVertex = RotVertex @ self.RotQ
                    self._replicate(RotVertex, LayersToDo - 1, "Vertex")

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