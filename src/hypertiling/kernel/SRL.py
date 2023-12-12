import numpy as np
import copy
import math

# relative imports
from .SR_base import KernelRotationalCommon
from .SRL_util import DuplicateContainerSimple
from ..arraytransformation import  mrotate, morigin
from ..distance import disk_distance


class StaticRotationalLegacy(KernelRotationalCommon):
    """ 
    Deprecated (!!)
    Our first tiling construction kernel, generates a hyperbolic lattice 
    by discrete rotations of existing polygons about their vertices
    """

    def __init__ (self, p, q, n, **kwargs):
        super(StaticRotationalLegacy, self).__init__(p, q, n, **kwargs)
        self.dgts = 10
        self.accuracy = 10**(-self.dgts) # numerical accuracy

        # construct tiling
        self.generate()


    def generate(self):
        """
        do full construction
        """
        self._generate_sector()
        self._replicate()


    def _generate_sector(self):
        """
        generates one p or q-fold sector of the lattice
        in order to avoid problems associated to rounding we construct the
        fundamental sector a little bit wider than 360/p degrees in filter
        out rotational duplicates after all layers have been constructed
        """

        # clear tiling
        self.polygons = []

        # add fundamental polygon to list
        self.fund_poly = self.create_fundamental_polygon()
                
        # tiling centered around vertex
        if self.center == 'vertex':

            # shift fundamental polygon such that one of its vertices is on the origin
            # if centered around a vertex, shift one vertex to origin
            morigin(self.p, self.fund_poly.verticesP[0], self.fund_poly.verticesP)
            vertangle = math.atan2(self.fund_poly.verticesP[1].imag, self.fund_poly.verticesP[1].real)
            mrotate(self.p, vertangle-self.mangle, self.fund_poly.verticesP)


        self.fund_poly_center = self.fund_poly.verticesP[self.p]
        self.polygons.append(self.fund_poly)

        # prepare sets which will contain the center coordinates
        # will be used for uniqueness checks
        dupl_large = DuplicateContainerSimple(self.dgts)
        dupl_small = DuplicateContainerSimple(self.dgts)
        dupl_large.add(self.fund_poly.centerP())

        # the actual construction
        self._populate_sector(dupl_large, dupl_small)



       