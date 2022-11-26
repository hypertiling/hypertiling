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
from ..util import fund_radius


class DuplicateContainer:
    # since set is a hashed type, we need to round

    def __init__(self, digits):
        self.digits = digits
        self.elements = set()

    def add(self, element):
        self.elements.add(np.round(element, self.digits))

    def is_duplicate(self, element):
        element = np.round(element, self.digits)
        return (element in self.elements)



class KernelStaticRotational(KernelRotationalCommon):
    """ Tiling construction algorithm written by M. Schrauth and F. Dusel  """

    def __init__ (self, p, q, n, center, autogenerate=True, radius=None):
        super(KernelStaticRotational, self).__init__(p, q, n, center, autogenerate, radius)
        self.dgts = 10
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
        dupl_large = DuplicateContainer(self.dgts)
        dupl_small = DuplicateContainer(self.dgts)
        dupl_large.add(self.fund_poly.centerP())

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

                                # add to duplicate container
                                dupl_large.add(center)

                                # create copy
                                polycopy = copy.deepcopy(pgon)

                                # generate adjacent polygon and add to large list
                                adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                                adj_pgon.layer = l + 1
                                self.polygons.append(adj_pgon)

                                # if angle is in slice, add to second duplicate container
                                if MANGLE <= cangle <= self.degtol + MANGLE:
                                    if not dupl_small.is_duplicate(center):
                                        dupl_small.add(center)

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

        # free mem of duplicate container
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



    def add_layer(self):
        """ constructs an additional layer for an existing tiling """

        newpolygons = []

        dupl_large = set()
        for pgon in self.polygons:
            center = np.round(pgon.centerP(), self.dgts)
            dupl_large.add(center)

        for pgon in self.polygons:
            # iterate over every vertex of pgon
            for vert_ind in range(self.p):
                # iterate over all polygons touching this very vertex
                for rot_ind in range(self.q):
                    # compute center and angle
                    center = mfull_point(pgon.verticesP[vert_ind], rot_ind * self.qhi, pgon.centerP())

                    cangle = math.degrees(math.atan2(center.imag, center.real))
                    cangle += 360 if cangle < 0 else 0

                    # try adding to centerlist; it is a set() and takes care of duplicates
                    lenA = len(dupl_large)
                    center = np.round(center, self.dgts)  # CAUTION
                    dupl_large.add(center)
                    lenB = len(dupl_large)

                    # this tells us whether an element has actually been added
                    if lenB > lenA:
                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon
                        adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                        adj_pgon.find_angle()

                        # add corresponding poly to large list
                        newpolygons.append(adj_pgon)

        self.polygons += newpolygons


        

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
