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
            centerset_extra = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]),
                                              math.atan2(self.fund_poly.verticesP[self.p].imag,
                                                         self.fund_poly.verticesP[self.p].real))
            centerarray = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]),
                                          math.atan2(self.fund_poly.verticesP[self.p].imag,
                                                     self.fund_poly.verticesP[self.p].real))
        else:
            centerset_extra = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]),
                                              self.phi / 2)  # the initial poly has a center of (0,0) therefore we set its angle artificially to phi/2
            centerarray = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]), self.phi / 2)
        # prepare sets which will contain the center coordinates
        # this is used for uniqueness checks later

        startpgon = 0
        endpgon = 1

        fr = fund_radius(self.p, self.q) / 2
        # loop over layers to be constructed
        for l in range(1, self.nlayers):

            # computes all neighbor polygons of layer l
            for pgon in self.polygons[startpgon:endpgon]:

                # iterate over every vertex of pgon
                for vert_ind in range(self.p):

                    # iterate over all polygons touching this very vertex
                    for rot_ind in range(self.q):
                        # compute center and angle
                        center = mfull_point(pgon.verticesP[vert_ind], rot_ind * self.qhi, pgon.verticesP[self.p])
                        cangle = math.degrees(math.atan2(center.imag, center.real))
                        cangle += 360 if cangle < 0 else 0

                        # cut away cells outside the fundamental sector
                        # allow some tolerance at the upper boundary
                        if (MANGLE <= cangle < sect_angle_deg + self.degtol + MANGLE) and (abs(center) > fr):
                            if not centerarray.fp_has(center):
                                centerarray.add(center)

                                # create copy
                                polycopy = copy.deepcopy(pgon)

                                # generate adjacent polygon
                                adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                                adj_pgon.layer = l + 1
                                # add corresponding poly to large list
                                self.polygons.append(adj_pgon)

                                # if angle is in slice, add to centerset_extra
                                if MANGLE <= cangle <= self.degtol + MANGLE:
                                    if not centerset_extra.fp_has(center):
                                        centerset_extra.add(center)

            startpgon = endpgon
            endpgon = len(self.polygons)

        # free mem of centerset
        del centerarray

        # filter out rotational duplicates
        deletelist = []

        for kk, pgon in enumerate(self.polygons):
            angle = math.degrees(math.atan2(pgon.verticesP[self.p].imag, pgon.verticesP[self.p].real))
            angle += 360 if angle < 0 else 0
            if angle > sect_angle_deg - self.degtol + MANGLE:
                center = moeb_rotate_trafo(-sect_angle, pgon.verticesP[self.p])
                if centerset_extra.fp_has(center):
                    deletelist.append(kk)
        self.polygons = list(np.delete(self.polygons, deletelist))

    def add_layer(self):
        """ constructs an additional layer for an existing tiling """

        newpolygons = []

        # prepare sets which will contain the center coordinates
        # this is used for uniqueness checks later
        if self.center == "vertex":

            centerarray = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]),
                                          math.atan2(self.fund_poly.verticesP[self.p].imag,
                                                     self.fund_poly.verticesP[self.p].real))
        else:
            centerarray = CenterContainer(self.p * self.q, abs(self.fund_poly.verticesP[self.p]), self.phi / 2)

        # fill the centerarray with already existing centers
        for pgon in tiling:
            center = np.round(pgon.centerP(), tiling.dgts)
            centerarray.add(center)

        for pgon in tiling:
            # iterate over every vertex of pgon
            for vert_ind in range(tiling.p):
                # iterate over all polygons touching this very vertex
                for rot_ind in range(tiling.q):
                    # compute center and angle
                    center = mfull_point(pgon.verticesP[vert_ind], rot_ind * self.qhi, pgon.verticesP[self.p])
                    cangle = math.degrees(math.atan2(center.imag, center.real))
                    cangle += 360 if cangle < 0 else 0
                    if not centerarray.fp_has(center):  # if it's a new polygon
                        centerarray.add(center)

                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon
                        adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                        adj_pgon.find_angle()

                        # add corresponding poly to large list
                        newpolygons.append(adj_pgon)

        tiling.polygons += newpolygons
