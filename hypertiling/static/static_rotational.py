import numpy as np
import math
import copy

# relative imports
from .static_base import KernelRotationalCommon
from ..arraytransformation import multi_rotation_around_vertex
from .static_rotational_util import DuplicateContainer


class KernelStaticRotational(KernelRotationalCommon):
    """
    High precision variant of the SR kernel, which uses a more sophisticated data container for duplicate checks
    """

    def __init__(self, p, q, n, center, autogenerate=True, radius=None):
        super(KernelStaticRotational, self).__init__(p, q, n, center, autogenerate, radius)
        
        # construct tiling
        if self.autogenerate:
            self.generate()


    def generate_sector(self):
        """
        generates one p or q-fold sector of the lattice
        in order to avoid problems associated to rounding we construct the
        fundamental sector a little bit wider than 360/p degrees in filter
        out rotational duplicates after all layers have been constructed
        """

        # clear tiling
        self.polygons = []

        # add fundamental polygon to list
        self.fund_poly = self.create_fundamental_polygon(self.center)
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
        self.populate_sector(dupl_large, dupl_small)
       



    def add_layer(self):
        """
        grow existing tiling outwards by one layer
        """

        newpolygons = []

        # new container for duplicate checks
        center = self.polygons[0].centerP()
        rrad = np.abs(center)
        pphi = math.atan2(center.imag, center.real)
        dupl_large = DuplicateContainer(self.p * self.q, rrad, pphi)
        
        # fill container
        for pgon in self.polygons:
            dupl_large.add(pgon.centerP())

        # loop over every polygon
        for pgon in self.polygons:

            # center of current polygon
            pgon_center = pgon.verticesP[self.p]

            # iterate over every vertex of pgon
            for vert_ind in range(self.p):

                # rotate polygon around current vertex
                # compute center coordinates of all polygons which share this vertex...
                adj_centers = multi_rotation_around_vertex(self.q, self.qhi, pgon.verticesP[vert_ind], pgon_center)            
                
                # ... and iterate over them
                for rot_ind in range(self.q):

                    center = adj_centers[rot_ind]

                    # check whether candidate polygon already exists
                    if not dupl_large.is_duplicate(center):

                        # add to duplicate container
                        dupl_large.add(center)

                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon and add to large list
                        adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                        newpolygons.append(adj_pgon)

        self.polygons += newpolygons
