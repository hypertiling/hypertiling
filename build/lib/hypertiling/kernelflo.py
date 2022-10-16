import numpy as np
import math
import copy

# relative imports
from .kernelbase import KernelCommon
from .hyperpolygon import HyperPolygon, mfull_point
from .transformation import moeb_rotate_trafo
from .util import fund_radius
from .CenterContainer import CenterContainer

class KernelFlo(KernelCommon):
    """
    High precision kernel written by F. Goth
    """
    def __init__ (self, p, q, n, center):
        super(KernelFlo, self).__init__(p, q, n, center)

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
        self.polygons.append(self.fund_poly)

        # angle width of the fundamental sector
        sect_angle     = self.phi
        sect_angle_deg = self.degphi
        if self.center == "vertex":
            sect_angle     = self.qhi
            sect_angle_deg = self.degqhi
            centerset_extra = CenterContainer(self.p*self.q, abs(self.fund_poly.centerP()), math.atan2(self.fund_poly.centerP().imag, self.fund_poly.centerP().real))
            centerarray = CenterContainer(self.p*self.q, abs(self.fund_poly.centerP()), math.atan2(self.fund_poly.centerP().imag, self.fund_poly.centerP().real))
        else:
            centerset_extra = CenterContainer(self.p*self.q, abs(self.fund_poly.centerP()), self.phi/2) # the initial poly has a center of (0,0) therefore we set its angle artificially to phi/2
            centerarray = CenterContainer(self.p*self.q, abs(self.fund_poly.centerP()), self.phi/2)
        # prepare sets which will contain the center coordinates
        # this is used for uniqueness checks later
    

        startpgon = 0
        endpgon = 1

        fr = fund_radius(self.p, self.q)/2
        # loop over layers to be constructed
        for l in range(1, self.nlayers):

            # computes all neighbor polygons of layer l
            for pgon in self.polygons[startpgon:endpgon]:

                # iterate over every vertex of pgon
                for vert_ind in range(self.p):

                    # iterate over all polygons touching this very vertex
                    for rot_ind in range(self.q):
                        # compute center and angle
                        center = mfull_point(pgon.verticesP[vert_ind], rot_ind*self.qhi, pgon.centerP())
                        cangle = math.degrees(math.atan2(center.imag, center.real))
                        cangle += 360 if cangle < 0 else 0

                        # cut away cells outside the fundamental sector
                        # allow some tolerance at the upper boundary
                        if (self.mangle <= cangle < sect_angle_deg+self.degtol+self.mangle) and (abs(center) > fr):
                            if not centerarray.fp_has(center):
                                centerarray.add(center)

                                # create copy
                                polycopy = copy.deepcopy(pgon)

                                # generate adjacent polygon
                                adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                                adj_pgon.find_angle()
                                adj_pgon.layer = l+1
                                # add corresponding poly to large list
                                self.polygons.append(adj_pgon)

                                # if angle is in slice, add to centerset_extra
                                if self.mangle <= cangle <= self.degtol+self.mangle:
                                    if not centerset_extra.fp_has(center):
                                        centerset_extra.add(center)

            startpgon = endpgon
            endpgon = len(self.polygons)

        # free mem of centerset
        del centerarray

        # filter out rotational duplicates
        deletelist = []

        for kk, pgon in enumerate(self.polygons):
            if pgon.angle > sect_angle_deg - self.degtol + self.mangle:
                center = moeb_rotate_trafo(pgon.centerP(), -sect_angle)
                if centerset_extra.fp_has(center):
                    deletelist.append(kk)
        self.polygons = list(np.delete(self.polygons, deletelist))
