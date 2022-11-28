import numpy as np
import math
import copy

# relative imports
from .static_base import KernelRotationalCommon
from .hyperpolygon import HyperPolygon
from ..transformation import moeb_rotate_trafo
from ..arraytransformation import mfull_point, multi_rotation_around_vertex
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



    def in_sector(self, z0):
        """
        Check whether point z0 is located in fundamental sector of the tiling
        """
        cangle = math.degrees(math.atan2(z0.imag, z0.real))
        if (self.sect_lbound <= cangle < self.sect_ubound) and (abs(z0) > self.fr2):
            return True
        else:
            return False

    def in_slice_lower(self, z0):
        """
        Check whether point z0 is located in lower soft boundary of fundamental sector
        This is required in order to check for rotational duplicates during the construction
        """
        cangle = math.degrees(math.atan2(z0.imag, z0.real))
        return cangle < self.lower_slice

    def in_slice_upper(self, z0):
        """
        Check whether point z0 is located in upper soft boundary of fundamental sector
        This is required in order to check for rotational duplicates during the construction
        """
        cangle = math.degrees(math.atan2(z0.imag, z0.real))
        return cangle > self.upper_slice





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

        # prepare sets which will contain the center coordinates
        # will be used for uniqueness checks
        dupl_large = DuplicateContainer(self.dgts)
        dupl_small = DuplicateContainer(self.dgts)
        dupl_large.add(self.fund_poly.centerP())

        startpgon = 0
        endpgon = 1

        # loop over layers to be constructed
        for l in range(1, self.nlayers):

            # computes all neighbor polygons of layer l
            for pgon in self.polygons[startpgon:endpgon]:

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

                        # check whether candidate polygon is in fundemantal sector
                        if self.in_sector(center):   

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

                                # if angle is in lower soft sector boundary, add to second duplicate container
                                if self.in_slice_lower(center): 
                                    if not dupl_small.is_duplicate(center):
                                        dupl_small.add(center)

            startpgon = endpgon
            endpgon = len(self.polygons)

            # check numerical stability before moving on to next layer
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
            center = pgon.verticesP[self.p]
            # if poly is inside soft boundary 
            # it has to be considered for rotational duplicate check
            if self.in_slice_upper(center): 
                # rotate center of poly back by sector angle
                center = moeb_rotate_trafo(-self.sect_angle, pgon.verticesP[self.p])
                # check whether we already have this rotated center
                # if so: rotational duplicate
                if dupl_small.is_duplicate(center):
                    # delete
                    deletelist.append(kk)

        # delete all rotational duplicates
        self.polygons = list(np.delete(self.polygons, deletelist))



    def add_layer(self):
        """
        grow existing tiling outwards by one layer
        """
        
        newpolygons = []

        # new container for duplicate checks
        dupl_large = DuplicateContainer(self.dgts)
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
