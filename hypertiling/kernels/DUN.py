import numpy as np
import copy
from ..ion import htprint
from .SR_base import KernelStaticBase
from .DUN_util import transformW_poly, transformW_site


# NOTE: This kernel implements the "original" construction algorithm of D. Dunham (1982)
# The algorithm uses Weierstraß (hyperboloid) coordinates; since those are not natively supported
# by our HyperPolygon class we need transformation functions provided in DUN_util.py


class KernelLegacyDunham(KernelStaticBase):
    """
    Original construction algorithm by D. Dunham (1982)
    works for every valid combination {p,q}
    however produces a lot of duplicates
    """

    def __init__ (self, p, q, n, **kwargs):
        super(KernelLegacyDunham, self).__init__(p, q, n, **kwargs)


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

        self.RotP = self.ReflectHypotenuse @ self.ReflectEdgeBisector
        self.RotQ = self.ReflectPgonEdge @ self.ReflectHypotenuse
        self.Rot2P = self.RotP @ self.RotP  # actually boosts the performance
        self.Rot3P = self.Rot2P @ self.RotP
        self.RotCenterG = np.eye(3)  # G for usage in generate()
        self.RotCenterR = np.eye(3)   # R for usage in replicate(...)

        # fundamental polygon of the tiling
        #self.fund_poly = self.create_fundamental_polygon(self.phi/2)
        self._create_first_layer(self.phi/2)

        # construct tiling
        self.generate()



    def generate(self):
        if self.n == 1:
            return

        for _ in range(self.p):
            RotVertex = self.RotCenterG @ self.RotQ
            self.replicate(self.polygons, RotVertex, self.n - 2, "Edge")
            for _ in range(self.q - 3):
                RotVertex = RotVertex @ self.RotQ
                self.replicate(self.polygons, RotVertex, self.n - 2, "Vertex")

            self.RotCenterG = self.RotCenterG @ self.RotP


    def replicate(self, Polygons, InitialTran, LayersToDo, AdjacencyType):
        poly = copy.deepcopy(self.fund_poly)
        #poly.transform(InitialTran)
        transformW_poly(poly,InitialTran)
        Polygons.append(poly)  # appending anything and removing duplicates afterwards is faster
        ExposedEdges = 0
        VertexPgons = 0

        if LayersToDo > 0:
            if AdjacencyType == "Edge":
                ExposedEdges = self.p - 3
                self.RotCenterR = InitialTran @ self.Rot3P
            if AdjacencyType == "Vertex":
                ExposedEdges = self.p - 2
                self.RotCenterR = InitialTran @ self.Rot2P

            for j in range(ExposedEdges):
                RotVertex = self.RotCenterR @ self.RotQ
                self.replicate(Polygons, RotVertex, LayersToDo - 1, "Edge")
                if j < ExposedEdges:  # I do not understand where -3 and -4 comes from
                    VertexPgons = self.q - 1  # was -3, corrected by trial and error
                elif j == ExposedEdges:
                    VertexPgons = self.q - 2  # and -4 in Dunhams paper

                for _ in range(VertexPgons):
                    RotVertex = RotVertex @ self.RotQ
                    self.replicate(Polygons, RotVertex, LayersToDo - 1, "Vertex")

                self.RotCenterR = self.RotCenterR @ self.RotP

    def add_layer(self):
        htprint("Warning", "The requested function is not implemented! Please use a different kernel!")
        return