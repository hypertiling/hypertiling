import numpy as np
import math
import copy

# relative imports
from .static_base import KernelRotationalCommon
from ..transformation import moeb_rotate_trafo
from ..arraytransformation import mfull_point
from ..util import fund_radius
from .static_rotational_improved_util import CenterContainer
from .static_base import MANGLE


class KernelStaticRotationalImproved(KernelRotationalCommon):
    """
    High precision kernel written by F. Goth
    """

    def __init__(self, p, q, n, center, autogenerate=True, radius=None):
        super(KernelStaticRotationalImproved, self).__init__(p, q, n, center, autogenerate, radius)
        
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

        # clear list
        self.polygons = []

        # add fundamental polygon to list
        self.fund_poly = self.create_fundamental_polygon(self.center)
        self.polygons.append(self.fund_poly)

        # angle width of the fundamental sector
        sect_angle = self.phi
        sect_angle_deg = self.degphi
        if self.center == "vertex":
            sect_angle = self.qhi
            sect_angle_deg = self.degqhi

            # prepare containers which will be used for uniqueness checks
            rrad = abs(self.fund_poly.verticesP[self.p])
            pphi = math.atan2(self.fund_poly.verticesP[self.p].imag, self.fund_poly.verticesP[self.p].real)

            dupl_small = CenterContainer(self.p * self.q, rrad, pphi)                
            dupl_large = CenterContainer(self.p * self.q, rrad, pphi)
        else:
            rrad = abs(self.fund_poly.verticesP[self.p])
            pphi = self.phi / 2
            # the initial poly has a center of (0,0) therefore we set its angle artificially to phi/2
            dupl_small = CenterContainer(self.p * self.q, rrad, pphi)
            dupl_large = CenterContainer(self.p * self.q, rrad, pphi)


        # half fundamental radius
        fr = fund_radius(self.p, self.q) / 2

        startpgon = 0
        endpgon = 1

        # loop over layers to be constructed
        for l in range(1, self.nlayers):

            # computes all neighbor polygons of layer l
            for pgon in self.polygons[startpgon:endpgon]:

                # iterate over every vertex of pgon
                for vert_ind in range(self.p):

                    # iterate over all polygons touching this very vertex
                    for rot_ind in range(self.q):
                        
                        # compute center and angle of the candidate
                        center = mfull_point(pgon.verticesP[vert_ind], rot_ind * self.qhi, pgon.verticesP[self.p])
                        cangle = math.degrees(math.atan2(center.imag, center.real))
                        cangle += 360 if cangle < 0 else 0

                        # cut away candidates outside the fundamental sector
                        # allow some tolerance at the upper boundary
                        sector_lbound = MANGLE
                        sector_ubound = sect_angle_deg + self.degtol + MANGLE
                        
                        if (sector_lbound <= cangle < sector_ubound) and (abs(center) > fr):
                            
                            # check whether candidate polygon already exists
                            if not dupl_large.is_duplicate(center):
                                                                
                                # add to center container
                                dupl_large.add(center)

                                # create copy
                                polycopy = copy.deepcopy(pgon)

                                # generate adjacent polygon and add to large list
                                adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                                adj_pgon.layer = l + 1
                                self.polygons.append(adj_pgon)

                                # if angle is in slice, add to centerset_extra
                                if MANGLE <= cangle <= self.degtol + MANGLE:
                                    if not dupl_small.is_duplicate(center):
                                        dupl_small.add(center)

            startpgon = endpgon
            endpgon = len(self.polygons)

        # free mem of centerset
        del dupl_large

        # --- filter out rotational duplicates
        deletelist = []
        
        # go through every polygon
        for kk, pgon in enumerate(self.polygons):
            # compute angle
            angle = math.degrees(math.atan2(pgon.verticesP[self.p].imag, pgon.verticesP[self.p].real))
            angle += 360 if angle < 0 else 0
            # if poly is inside soft boundary 
            # it has to be considered for rotational duplicate check
            if angle > MANGLE + sect_angle_deg - self.degtol :
                # rotate center of poly back by sector angle
                center = moeb_rotate_trafo(-sect_angle, pgon.verticesP[self.p])
                # check whether we already have this rotated center
                # if so: rotational duplicate
                if dupl_small.is_duplicate(center):
                    # delete
                    deletelist.append(kk)
        # delete all rotational duplicates
        self.polygons = list(np.delete(self.polygons, deletelist))


    def is_duplicate(self, center, dupl_large):
        return dupl_large.fp_has(center)


    def add_layer(self):
        """ constructs an additional layer for an existing tiling """

        newpolygons = []

        # prepare sets which will contain the center coordinates
        # this is used for uniqueness checks later
        if self.center == "vertex":

            dupl_large = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]),
                                          math.atan2(self.fund_poly.verticesP[self.p].imag,
                                                     self.fund_poly.verticesP[self.p].real))
        else:
            dupl_large = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]), self.phi / 2)

        # fill the dupl_large with already existing centers
        for pgon in self.polygons:
            center = np.round(pgon.centerP(), self.dgts)
            dupl_large.add(center)

        for pgon in self.polygons:
            # iterate over every vertex of pgon
            for vert_ind in range(self.p):
                # iterate over all polygons touching this very vertex
                for rot_ind in range(self.q):
                    # compute center and angle
                    center = mfull_point(pgon.verticesP[vert_ind], rot_ind * self.qhi, pgon.verticesP[self.p])
                    cangle = math.degrees(math.atan2(center.imag, center.real))
                    cangle += 360 if cangle < 0 else 0
                    if not dupl_large.fp_has(center):  # if it's a new polygon
                        dupl_large.add(center)

                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon
                        adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                        adj_pgon.find_angle()

                        # add corresponding poly to large list
                        newpolygons.append(adj_pgon)

        self.polygons += newpolygons
