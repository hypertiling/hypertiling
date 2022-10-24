import numpy as np
import math
import copy

# relative imports
from .static_base import KernelRotationalCommon
from .hyperpolygon import HyperPolygon
from ..transformation import moeb_rotate_trafo
from ..arraytransformation import mfull_point
from ..distance import disk_distance
from .static_base import MANGLE

class KernelStaticRotational(KernelRotationalCommon):
    """ Tiling construction algorithm written by M. Schrauth and F. Dusel  """

    def __init__ (self, p, q, n, center, autogenerate=True, radius=None):
        super(KernelStaticRotational, self).__init__(p, q, n, center, autogenerate, radius)
        #self.center = center # automatically assigned from super class?
        self.dgts = 8
        self.accuracy = 10**(-self.dgts) # numerical accuracy

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
        sect_angle     = self.phi
        sect_angle_deg = self.degphi
        if self.center == "vertex":
            sect_angle     = self.qhi
            sect_angle_deg = self.degqhi

        # prepare sets which will contain the center coordinates
        # will be used for uniqueness checks
        centerset = set()
        centerset_extra = set()
        centerset.add(np.round(self.fund_poly.centerP(), self.dgts))

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
                        # compute center and angle
                        center = mfull_point(pgon.verticesP[vert_ind], rot_ind*self.qhi, pgon.centerP())
                        
                        cangle = math.degrees(math.atan2(center.imag, center.real))
                        cangle += 360 if cangle < 0 else 0

                        # cut away cells outside the fundamental sector
                        # allow some tolerance at the upper boundary
                        if MANGLE-1e-14 <= cangle < sect_angle_deg+self.degtol+MANGLE:

                            # try adding to centerlist; it is a set() and takes care of duplicates
                            center = np.round(center, self.dgts)
                            lenA = len(centerset)
                            centerset.add(center)
                            lenB = len(centerset)

                            # this tells us whether an element has actually been added
                            if lenB>lenA:
                                # create copy
                                polycopy = copy.deepcopy(pgon)

                                # generate adjacent polygon
                                adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                                adj_pgon.find_angle()
                                adj_pgon.layer = l+1

                                # add corresponding poly to large list
                                self.polygons.append(adj_pgon)

                                # if angle is in slice, add to centerset_extra
                                if MANGLE-1e-14 <= cangle <= self.degtol+MANGLE:
                                    centerset_extra.add(center)

            startpgon = endpgon
            endpgon = len(self.polygons)

            if self.numerically_unstable_upper(l, startpgon, endpgon):
                print("Numerical accuracy exhausted;")
                print("No more layers will be constructed; automatic shutdown")
                break

            if self.numerically_unstable_lower(l, startpgon, endpgon):
                print("Accumulated numerical errors have become too large;")
                print("No more layers will be constructed; automatic shutdown")
                break

        # free mem of centerset
        del centerset

        # filter out rotational duplicates
        deletelist = []
        for kk, pgon in enumerate(self.polygons):
            if pgon.angle > sect_angle_deg-self.degtol+MANGLE:

                center = moeb_rotate_trafo(-sect_angle, pgon.centerP())

                center = np.round(center, self.dgts) # better use simple distance?

                if center in centerset_extra:
                    deletelist.append(kk)

        self.polygons = list(np.delete(self.polygons, deletelist))


    def add_layer(self):
        """ constructs an additional layer for an existing tiling """

        newpolygons = []

        centerset = set()
        for pgon in tiling:
            center = np.round(pgon.centerP(), tiling.dgts)
            centerset.add(center)

        for pgon in tiling:
            # iterate over every vertex of pgon
            for vert_ind in range(tiling.p):
                # iterate over all polygons touching this very vertex
                for rot_ind in range(tiling.q):
                    # compute center and angle
                    center = mfull_point(pgon.verticesP[vert_ind], rot_ind * tiling.qhi, pgon.centerP())

                    cangle = math.degrees(math.atan2(center.imag, center.real))
                    cangle += 360 if cangle < 0 else 0

                    # cut away cells outside the fundamental sector
                    # allow some tolerance at the upper boundary
                    # try adding to centerlist; it is a set() and takes care of duplicates
                    lenA = len(centerset)
                    center = np.round(center, tiling.dgts)  # CAUTION
                    centerset.add(center)
                    lenB = len(centerset)

                    # this tells us whether an element has actually been added
                    if lenB > lenA:
                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon
                        adj_pgon = tiling.generate_adj_poly(polycopy, vert_ind, rot_ind)
                        adj_pgon.find_angle()

                        # add corresponding poly to large list
                        newpolygons.append(adj_pgon)

        tiling.polygons += newpolygons


        

    def numerically_unstable_upper(self, l, start, end, tolfactor=10, samplesize=10):
        """
        check whether the true "embedding" distance between cells in layer l comes close
        to the rounding accuracy
        """

        # innermost layers are always fine, do nothing
        if l<3:
            return False

        # randomly pick a number of sites from l-th layer
        curr_layer = self.polygons[start:end]
        layersize = end-start
        true_dists = []

        for i in range(samplesize):
            rndidx = np.random.randint(layersize)

            # generate an adjacent cell
            mother = curr_layer[rndidx]
            child  = self.generate_adj_poly(copy.deepcopy(mother), 0, 1)

            # compute the true (non-geodesic) distance
            true_dist = np.abs(mother.centerP()-child.centerP())
            true_dists.append(true_dist)

        # if this distances comes close to the rounding accuracy
        # two cells can no longer be reliably distinguished
        if np.min(true_dist) < self.accuracy*tolfactor:
            return True
        else:
            return False


    def numerically_unstable_lower(self, l, start, end, tolfactor=10, samplesize=100):
        """
        we know which geodesic distance two adjancent cells are supposed to have;
        here we take a sample of cells from the l-th layer and compute mutual 
        distances; if one of those is significantly off compared to the expected
        value we are about to enter a dangerous regime in terms of rounding errors
        """

        # innermost layers are always fine, do nothing
        if l<3:
            return False

        # take a sample of cells and compute their distances
        samples = self.polygons[start:end][:samplesize]
        disk_distances = []
        for j1, pgon1 in enumerate(samples):
            for j2, pgon2 in enumerate(samples):
                if j1 != j2:
                    disk_distances.append(disk_distance(pgon1.centerP(), pgon2.centerP()))

        # we are interested in the minimal distance (can be interpreted as an 
        # upper bound on the accumulated error)
        mindist = np.min(np.array(disk_distances))

        # the reference distance
        refdist = disk_distance(self.fund_poly.centerP(), self.polygons[1].centerP())

        # if out arithmetics worked error-free, mindist = refdist
        # in practice, it does not, so we compute the difference
        # if it comes close to the rounding accuracy, adjacency can no longer
        # by reliably resolved and we are about to enter a possibly unstable regime
        if np.abs(mindist-refdist) > self.accuracy/tolfactor:
            return True
        else:
            return False
