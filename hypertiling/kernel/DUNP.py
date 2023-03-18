import numpy as np
import copy
from ..ion import htprint
from .SR_base import KernelStaticBase
from .DUN_util import transformW_poly, transformW_site
from scipy.stats import circmean


# a transformation matrix with some extra information
class DunhamTransformation:
    """
    Transformations contain a matrix, the orientation, 
    and an index, pPosition, of the edge across which 
    the last transformation was made
    """
    def __init__(self, matrix, orientation, p_position):
        self.matrix = matrix
        self.orientation = orientation
        self.p_position = p_position

    def __mul__(self, other):
        # 
        new_matrix = self.matrix @ other.matrix
        new_orient = self.orientation * other.orientation
        new_p_pos  = other.p_position

        return DunhamTransformation(new_matrix, new_orient, new_p_pos)


def trafoW(xyts, trafo):
    # Apply Weierstrass transformation matrix
    return [trafo@k for k in xyts]


def rotationW(phi):
    # return Weierstrass rotation matrix
    return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])


class LegacyDunhamPlus(KernelStaticBase):
    """
    """

    def __init__ (self, p, q, n, **kwargs):
        super(LegacyDunhamPlus, self).__init__(p, q, n, **kwargs)

        if p==3 or q==3:
            htprint("Warning", "p=3 or q=3 not supported")

        if self.center == "vertex":
            htprint("Warning", "Dunham kernel does not support vertex centered tilings yet!")

        # reflection and rotation matrices
        self.b = np.arccosh(np.cos(np.pi / q) / np.sin(np.pi / p))

        self.ReflectPgonEdge = np.array([[-np.cosh(2 * self.b), 0, np.sinh(2 * self.b)],
                                         [0, 1, 0],
                                         [-np.sinh(2 * self.b), 0, np.cosh(2 * self.b)]])


        # We define the exposure of a p-gon in terms of the number of edges 
        # it has in common with the next layer.
        # A p-gon has minimum exposure if it has the fewest edges in common with 
        # the next layer, and thus shares an edge with the previous layer.
        # A p-gon has maximum exposure if it has the most edges in common with the 
        # next layer, and thus only shares a vertex with the previous layer.
        # We abbreviate these values as min_exp and max_exp, respectively.

        self.max_exp = self.p - 2
        self.min_exp = self.p - 3


        # fundamental polygon of the tiling
        self._create_first_layer(self.phi/2)

        # A tiling pattern is determined by how the p-gon pattern is
        # transformed across p-gon edges. These transformations are given here (TODO)
        self.edge_tran = self._compute_edge_reflections()

        # construct tiling
        self._generate()


    def _compute_edge_reflections(self):

        trafos = []

        for i in range(self.p):
            j = int((i+1)%self.p)

            phi1 = np.angle(self.fund_poly.verticesP[i])
            phi2 = np.angle(self.fund_poly.verticesP[j])
            phi = circmean([phi1,phi2])

            edge_trafo = rotationW(-phi) @ self.ReflectPgonEdge @ rotationW(phi)

            trafos.append(DunhamTransformation(edge_trafo, -1, i))
        
        return trafos

        


    # add polygon to tiling
    def _draw_pgon_pattern(self, trans):
        # create permanent copy of fundamental polygon
        poly = copy.deepcopy(self.fund_poly)
        # apply transformation
        transformW_poly(poly, trans.matrix)
        # draw, i.e. add to list
        self.polygons.append(poly)


    # increment transformation
    def add_to_tran(self, tran, shift):
        if shift % self.p == 0:
            return tran
        else:
            return self.compute_tran(tran, shift)

    def compute_tran(self, tran, shift):
        newEdge = (tran.p_position + tran.orientation * shift) % self.p
        return tran*self.edge_tran[newEdge]


    # central recursion
    def replicate_motif(self, poly, initialTran, layer, exposure):
        
        # draw polygon
        self._draw_pgon_pattern(initialTran)

        # proceed to desired depth
        if layer < self.n:
            #  Which vertex to start at
            min_exposure = (exposure == self.min_exp)
            pShift = 1 if min_exposure else 0
            verticesToDo = self.p-3 if min_exposure else self.p-2

            # Iterate over vertices
            for i in range(1, verticesToDo+1):
                first_i = (i==1)
                pTran = self.compute_tran(initialTran, pShift)
                qSkip = -1 if first_i else 0
                qTran = self.add_to_tran(pTran, qSkip)
                pgonsToDo = self.q-3 if first_i else self.q-2

                # Iterate about a vertex
                for j in range(1, pgonsToDo+1):
                    first_j = (j==1)
                    newExposure = self.min_exp if first_j else self.max_exp
                    self.replicate_motif(poly, qTran, layer+1, newExposure)
                    qTran = self.add_to_tran(qTran, -1)

                # Advance to next vertex
                pShift = (pShift + 1) % self.p
            


    # top-level driver routine
    def replicate(self, poly):
        # the fundamental polygon itself has already been
        # added to the tiling by the init method
        
        # Iterate over each vertex
        for i in range(1, self.p+1):
            qTran = self.edge_tran[i-1]

            # Iterate about a vertex
            for j in range(1, self.q-2+1):
                exposure = self.min_exp if (j==1) else self.max_exp
                self.replicate_motif(poly, qTran, 2, exposure)
                qTran = self.add_to_tran(qTran, -1)



    def _generate(self):
        self.replicate(self.fund_poly)