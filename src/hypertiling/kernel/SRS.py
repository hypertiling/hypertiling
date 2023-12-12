import numpy as np
import math
from ..arraytransformation import morigin, mrotate
from .SR_util import DuplicateContainer
from .SR_base import KernelRotationalCommon



class StaticRotationalSector(KernelRotationalCommon):
    """
    Static Rotational Sector (SRS) kernel

    New cells are constructed in a semi-brute force way via rotations about vertices of existing ones.
    Duplicates are eliminated using specialized data containers

    This is a variant of the default SRG kernel, where the lattice is only explicetly constructed in one
    symmetry sector and copied to the remaining sectors
    """

    def __init__(self, p, q, n, **kwargs):
        super(StaticRotationalSector, self).__init__(p, q, n, **kwargs)   

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

        # prepare container which will be used for duplicate checks
        if self.center == "vertex":
            rrad = np.abs(self.fund_poly_center)
            pphi = math.atan2(self.fund_poly_center.imag, self.fund_poly_center.real)
        if self.center == "cell":
            # the initial poly has a center of (0,0) 
            # therefore we set its angle artificially to phi/2
            rrad = 0
            pphi = self.phi / 2
            
        
        # container used for filtering duplicates in the bulk
        dupl_large = DuplicateContainer(self.p * self.q, rrad, pphi)

        # container used for filtering rotational duplicates at the sector boundary
        dupl_small = DuplicateContainer(self.p * self.q, rrad, pphi)

        # the actual construction
        self._populate_sector(dupl_large, dupl_small)
       


