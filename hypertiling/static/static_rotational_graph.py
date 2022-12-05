import numpy as np
import math
import copy
from .static_base import KernelStaticBase, KernelRotationalCommon
from ..arraytransformation import mfull, mrotate, morigin, multi_rotation_around_vertex
from ..util import fund_radius
from ..ion import htprint
from ..arraytransformation import multi_rotation_around_vertex
from .static_rotational_graph_util import DuplicateContainerCircular

PI2 = 2 * np.pi

# Magic number: transcendental number (Champernowne constant)
# used as an angular offset, rotates the entire construction by a bit during construction
MAGICANGLE = np.radians(0.1234567891011121314151617181920212223242526272829303132333)


class KernelStaticRotationalGraph(KernelRotationalCommon):
    """
    Hyperbolic tiling construction kernel

    unlike the other static rotational kernels, here the neighbours are computed upon construction of the tiling
    however, since currently no sector algorithm is used, the construction itself is slower
    """
    def __init__ (self, p, q, n, center, autogenerate=True, radius=None):
        super(KernelStaticRotationalGraph, self).__init__(p, q, n, center, autogenerate, radius)


        # prepare list to store neighours
        self.nbrs = []

        # some variables
        self.layers = 1
        self.outmost_layer_lower = 0
        self.outmost_layer_upper = 1

        # construct tiling
        if self.autogenerate:
            self.generate()



    def get_nbrs(self, i):
        """
        return neighbours of cell i as list
        """
        return self.nbrs[i]


    def add_layer(self, filter=None):
        """
        add layer
        """

        self.layers += 1

        if filter is None:
            filter = self.not_origin

        # computes all neighbor polygons of layer l
        for pgon in self.polygons[self.outmost_layer_lower:self.outmost_layer_upper]:

            # center of current polygon
            pgon_center = pgon.verticesP[self.p]

            collect_nbrs = []
            
            # iterate over every vertex of pgon
            for vert_ind in range(self.p):

                # rotate polygon around current vertex
                # compute center coordinates of all polygons which share this vertex...
                adj_centers = multi_rotation_around_vertex(self.q, self.qhi, pgon.verticesP[vert_ind], pgon_center)            
                
                # ... and iterate over them
                for rot_ind in range(self.q):

                    center = adj_centers[rot_ind]

                    # check whether candidate polygon is in fundemantal sector
                    if filter(center):   

                        # check whether candidate polygon already exists
                        duplicate, idx = self.dplcts.is_duplicate(center)
                        if not duplicate:

                            # add to duplicate container
                            self.dplcts.add(center,len(self.polygons))

                            # create copy
                            polycopy = copy.deepcopy(pgon)

                            # generate adjacent polygon and add to large list
                            collect_nbrs.append(len(self.polygons))
                            adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                            adj_pgon.layer = self.layers
                            self.polygons.append(adj_pgon)

                        else:
                            collect_nbrs.append(idx)
                    else:
                        collect_nbrs.append(0)

            collect_nbrs = np.array(collect_nbrs)
            collect_nbrs = collect_nbrs[collect_nbrs != self.counter]
            self.nbrs.append(list(np.unique(collect_nbrs)))
            self.counter += 1


        self.outmost_layer_lower = self.outmost_layer_upper
        self.outmost_layer_upper = len(self.polygons)

        #htprint("Status", "Created a new layer with index", self.layers+1, "containing", self.outerlayer_lower-self.outerlayer_upper, "polygons")



    def _prepare_duplicate_container(self):
        """
        prepare containers for duplicate checks
        init with origin, set angle artificially to phi/2
        """

        idx = 0
        rrad = 0
        pphi = self.phi / 2
        self.dplcts = DuplicateContainerCircular(self.p * self.q, rrad, pphi, idx)        

        # add full first layer for vertex centered tilings
        if self.center == "vertex":
            for i in range(0,self.q):
                self.dplcts.add(self.polygons[i].centerP(),i)

        # current layer number
        self.layers = 1




    def generate(self):
        """
        do full construction
        """


        # clear tiling
        self.polygons = []

        # add layers repeatedly until nlayers is reached
        if self.center == "cell":
            self._create_first_layer()
            self._prepare_duplicate_container()
            for _ in range(self.nlayers-1):
                self.add_layer(self.not_origin)

        elif self.center == "vertex":

            self._create_first_layer()
            self._prepare_duplicate_container()
            for _ in range(self.nlayers-1):
                self.add_layer(self.filter_always_pass)



    def generate_adj_poly(self, polygon, ind, k):
        """
        finds the next polygon by k-fold rotation of polygon around the vertex number ind
        """
        mfull(self.p, k * self.qhi, ind, polygon.verticesP)
        return polygon


    def filter_always_pass(self, z0):
        return True

    def in_sector(self, z0):
        """
        Check whether point z0 is located in fundamental sector of the tiling
        """
        cangle = math.atan2(z0.imag, z0.real)
        if (self.sect_lbound <= cangle < self.sect_ubound) and (abs(z0) > self.fr2):
            return True
        else:
            return False


    def not_origin(self, z0):
        """
        Check whether point z0 is located at the origin
        """
        return (abs(z0) > self.fr2)



# ------------- Transformations -------------


    def rotate(self, angle: float, deg=False):
        """
        Rotates the whole tiling about the origin.
        
        Parameters
        ----------
        angle: float
            Angle in radians by which the tiling is rotated.
        
        deg: bool, default: False
            If True, then angle is considered in units of degrees.
        """

        if deg:
            angle = angle * math.pi / 180

        for poly in self.polygons:
            mrotate(self.p, angle, poly.verticesP)

    def translate(self, z):
        """ 
        Translates the whole tiling so that the point z lays in the origin.
        
        Parameters
        ----------
        z: complex
            The point which will be translated to the origin.
        """

        for poly in self.polygons:
            morigin(self.p, z, poly.verticesP)


