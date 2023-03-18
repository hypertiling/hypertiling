import numpy as np
import copy
from ..ion import htprint
from .SR_base import KernelStaticBase
from .DUN_util import transformW_poly, transformW_site


class DunTrafo:
    def __init__(self, matrix, orientation, p_position):
        self.matrix = matrix
        self.orientation = orientation
        self.p_position = p_position

    def __mult__(self, other):
        return self.matrix @ other.matrix
        


class LegacyDunhamPlus(KernelStaticBase):
    """
    This kernel implements the "original" construction algorithm of D. Dunham (1982)
    The algorithm uses Weierstraß (hyperboloid) coordinates; since those are not natively supported
    by our HyperPolygon class we need transformation functions provided in DUN_util.py
    """

    def __init__ (self, p, q, n, **kwargs):
        super(LegacyDunhamPlus, self).__init__(p, q, n, **kwargs)


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

        self.max_exp = self.p - 2
        self.min_exp = self.p - 3


        self.vertex_valences = self.q * np.ones(self.p).astype("int")


        self.edge_tran = [self.ReflectPgonEdge,
                          self.ReflectPgonEdge@self.RotP, 
                          self.ReflectPgonEdge@self.Rot2P,
                          self.ReflectPgonEdge@self.Rot3P,
                          self.ReflectPgonEdge@self.Rot3P@self.RotP,
                          self.ReflectPgonEdge@self.Rot3P@self.Rot2P,
                          self.ReflectPgonEdge@self.Rot3P@self.Rot3P]
        

        self.pShiftArray = [1, 0]
        self.verticesToSkipArray = [3, 2]
        self.qShiftArray = [0, -1]
        self.polygonsToSkipArray = [2, 3]
        self.exposureArray = [self.max_exp, self.min_exp]

        # fundamental polygon of the tiling
        self._create_first_layer(self.phi/2)

        # construct tiling
        self._generate()


    def _draw_pgon_pattern(self, trans):
        # create permanent copy of fundamental polygon
        poly = copy.deepcopy(self.fund_poly)
        # apply transformation
        transformW_poly(poly, trans.matrix)
        # draw, i.e. add to list
        self.polygons.append(poly)



    def add_to_tran(self, tran, shift):
        if shift % self.p == 0:
            return tran
        else:
            return self.compute_tran(tran, shift)
    

    def compute_tran(self, tran, shift):
        newEdge = (tran.p_position + tran.orientation * shift) % self.p
        return tran*self.edge_tran[newEdge]



    def replicate_motif(self, poly, initialTran, layer, exposure):
        self._draw_pgon_pattern(initialTran)

        if layer <= self.n:
            #  Which vertex to start at
            pShift = self.pShiftArray[exposure]
            verticesToDo = self.p - self.verticesToSkipArray[exposure]

            # Iterate over vertices
            for i in range(1, verticesToDo+1):
                pTran = self.compute_tran(initialTran, pShift)
                first_i = (i==1)
                qTran = self.add_to_tran(initialTran, pShift[first_i])

                if (pTran.orientation > 0):
                    vertex = (pTran.p_position-1) % self.p
                else: 
                    vertex = pTran.p_position

                polygonsToDo = self.vertex_valences[vertex] - self.polygonsToSkipArray[first_i]

                # Iterate about a vertex
                for j in range(1, polygonsToDo):
                    first_j = (j==1)
                    newExposure = self.exposureArray[first_j]

                    self.replicate_motif(poly, qTran, layer+1, newExposure)

                    qTran = self.add_to_tran(qTran, -1)

                # Advance to next vertex
                pShift = (pShift + 1) % self.p
            


    def replicate(self, poly):
        qtran = DunTrafo(self.edge_tran[1], 0, 1)
        
        

        for j in range(0, self.vertex_valences[1]):
            self.replicate_motif(poly, qtran, 2, self.max_exp)
            qtran = self.add_to_tran(qtran, -1)


    def _generate(self):
        self.replicate(self.fund_poly)