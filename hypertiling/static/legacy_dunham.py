import numpy as np
import copy

# relative imports
from .static_base import KernelStaticBase
from .hyperpolygon import HyperPolygon

from ..representations import p2w_xyt, w2p_xyt

# NOTE: This kernel implements the "original" construction algorithm of D. Dunham (1982)
# The algorithm uses Weierstraß (hyperboloid) coordinates; since those are not natively supported
# by our HyperPolygon class we need the following two transformation functions:

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


class KernelLegacyDunham(KernelStaticBase):
    """
    Original construction algorithm by D. Dunham (1982)
    works for every valid combination {p,q}
    however produces a lot of duplicates
    """

    def __init__ (self, p, q, n, center="cell", autogenerate=True):
        super(KernelLegacyDunham, self).__init__(p, q, n, center, autogenerate)

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
        self.fund_poly = self.create_fundamental_polygon(center, rotate_by=360/p/2)

        # construct tiling
        if self.autogenerate:
            self.generate()


        
        

    # def create_fundamental_polygon(self):  # constructs the verticesP of the fundamental hyperbolic {p,q} polygon
    #     r = fund_radius(self.p, self.q)
    #     polygon = HyperPolygon(self.p)
    #     angle = np.pi / self.p
    #     for i in range(self.p):  # for every corner of the polygon
    #         z = complex(r * np.cos(angle + 2 * np.pi * i / self.p), r * np.sin(angle + 2 * np.pi * i / self.p))
    #         polygon.verticesP[i] = z
    #         polygon.verticesW[:, i] = p2w(z)
        #return polygon

    def generate(self):
        if self.nlayers == 1:
            return

        for _ in range(self.p):
            RotVertex = self.RotCenterG @ self.RotQ
            self.replicate(self.polygons, RotVertex, self.nlayers - 2, "Edge")
            for _ in range(self.q - 3):
                RotVertex = RotVertex @ self.RotQ
                self.replicate(self.polygons, RotVertex, self.nlayers - 2, "Vertex")

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
        print('[hypertiling]: Error: The requested function is not implemented! Please use a different kernel!')
        return