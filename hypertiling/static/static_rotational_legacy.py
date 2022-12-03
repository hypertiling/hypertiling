import numpy as np
import copy
import math

# relative imports
from .static_base import KernelRotationalCommon, MAGICANGLE
from .static_rotational_legacy_util import DuplicateContainerSimple
from ..arraytransformation import  mrotate, morigin
from ..distance import disk_distance


class KernelStaticRotationalLegacy(KernelRotationalCommon):
    """ 
    A generic tiling construction kernel, generates a hyperbolic lattice 
    by discrete rotations of existing polygons about their vertices 
    """

    def __init__ (self, p, q, n, center, autogenerate=True, radius=None):
        super(KernelStaticRotationalLegacy, self).__init__(p, q, n, center, autogenerate, radius)
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
            mrotate(self.p, vertangle-MAGICANGLE, self.fund_poly.verticesP)


        self.fund_poly_center = self.fund_poly.verticesP[self.p]
        self.polygons.append(self.fund_poly)

        # prepare sets which will contain the center coordinates
        # will be used for uniqueness checks
        dupl_large = DuplicateContainerSimple(self.dgts)
        dupl_small = DuplicateContainerSimple(self.dgts)
        dupl_large.add(self.fund_poly.centerP())

        # the actual construction
        self.populate_sector(dupl_large, dupl_small)



        

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

