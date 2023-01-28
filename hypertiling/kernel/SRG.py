import numpy as np
import math
import copy
from .SR_base import KernelRotationalCommon
from ..arraytransformation import mfull, mrotate, morigin, multi_rotation_around_vertex
from ..arraytransformation import multi_rotation_around_vertex
from .SRG_util import DuplicateContainerCircular

PI2 = 2 * np.pi


class StaticRotationalGraph(KernelRotationalCommon):
    """
    Hyperbolic tiling construction kernel

    unlike the other static rotational kernels, here the neighbours are computed upon construction of the tiling
    however, since currently no sector algorithm is used, the construction itself is slower
    """
    def __init__ (self, p, q, n, **kwargs):
        super(StaticRotationalGraph, self).__init__(p, q, n, **kwargs)

        # define type of neighbour container
        self._nbrs = []

        # some variables
        self.layers = 1
        self.outmost_layer_lower = 0
        self.outmost_layer_upper = 1

        self.layerbounds = [0]
        self.closed = False

        # construct tiling
        self.generate()



    def find_poly_by_idx(self, idx):
        """
        scan through self.polygons and find the polygon with attribute idx=idx
        """
        for i,poly in enumerate(self.polygons):
            if poly.idx == idx:
                return i



    def remove_cells(self, deletelist):
        """
        deletelist : List[int]
            list of polygon indices to be removed
            note that this is not their index (position) in the array, but their HyperPolygon.idx!
            this routine automatically takes care of finding the correct position, though
        """

        if self.closed == False:
            errstr1 = "[hypertiling] Error: You must close the boundary of the lattice before removing cells!"
            errstr2 = "This can be done by calling either add_ghost_boundary or add_full_boundary (not implemented)"
            raise Exception(errstr1+"\n"+errstr2)
        
        # extract indices of cells
        idxlst = [poly.idx for poly in self.polygons]

        # index of cell in self.polygons und self._nbrs
        positions = [i for i in range(len(idxlst)) if idxlst[i] in deletelist]

        # remove all appearances of the index in the neighbour lists
        # of other cells
        for j,pos in enumerate(positions):
            for nb in self._nbrs[pos]:
                try:
                    self._nbrs[self.find_poly_by_idx(nb)].remove(deletelist[j])

                except:
                    pass

            
        # remove entry in polygons list and neighbours list    
        for index in sorted(positions, reverse=True):
            del self.polygons[index]
            del self._nbrs[index]



    def add_layer(self, filter=None):
        """
        add layer
        """
        self.closed = False

        self.layers += 1

        if filter is None:
            filter = self.not_origin

        # computes all neighbor polygons of layer l
        for pgon in self.polygons[self.outmost_layer_lower:self.outmost_layer_upper]:

            # center of current polygon
            pgon_center = pgon.verticesP[self.p]

            collect_nbrs = []
            
            # iterate over every vertex of pgon
            for vert_ind in reversed(range(self.p)):

                # rotate polygon around current vertex
                # compute center coordinates of all polygons which share this vertex...
                adj_centers = multi_rotation_around_vertex(self.q, self.qhi, pgon.verticesP[vert_ind], pgon_center)            
                
                # ... and iterate over them
                for rot_ind in range(self.q):

                    center = adj_centers[rot_ind]

                    # check whether candidate polygon is not closed to the origin
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
                            adj_pgon.idx = len(self.polygons)
                            self.polygons.append(adj_pgon)

                        else:
                            collect_nbrs.append(idx)
                    else:
                        collect_nbrs.append(0)

            collect_nbrs = np.array(collect_nbrs)
            collect_nbrs = collect_nbrs[collect_nbrs != self.counter]
            self._nbrs.append(list(np.unique(collect_nbrs)))
            self.counter += 1

        self.outmost_layer_lower = self.outmost_layer_upper
        self.outmost_layer_upper = len(self.polygons)
        self.layerbounds.append(self.outmost_layer_lower)



    def add_vertices(self, addlist, filter=None):
        """
        add layer
        """
        self.closed = False

        # extract indices of cells
        idxlst = [poly.idx for poly in self.polygons]

        # index of cell in self.polygons und self._nbrs
        positions = [i for i in range(len(idxlst)) if idxlst[i] in addlist]

        print(positions)

        self.layers += 1

        if filter is None:
            filter = self.not_origin

        # computes all neighbor polygons of layer l
        for idx in positions:
            pgon = self.polygons[idx]

            # center of current polygon
            pgon_center = pgon.verticesP[self.p]

            collect_nbrs = []
            
            # iterate over every vertex of pgon
            for vert_ind in reversed(range(self.p)):

                # rotate polygon around current vertex
                # compute center coordinates of all polygons which share this vertex...
                adj_centers = multi_rotation_around_vertex(self.q, self.qhi, pgon.verticesP[vert_ind], pgon_center)            
                
                # ... and iterate over them
                for rot_ind in range(self.q):

                    center = adj_centers[rot_ind]

                    # check whether candidate polygon is not closed to the origin
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
                            adj_pgon.idx = len(self.polygons)
                            self.polygons.append(adj_pgon)

                        else:
                            collect_nbrs.append(idx)
                    else:
                        collect_nbrs.append(0)

            collect_nbrs = np.array(collect_nbrs)
            collect_nbrs = collect_nbrs[collect_nbrs != self.counter]
            self._nbrs.append(list(np.unique(collect_nbrs)))
            self.counter += 1



    def add_ghost_boundary(self):
        """
        the construction principle of SRG does not allow to determine the neighbours of the outmost layer points
        during generation of the lattice

        calling this function, they will at least be connected to their parents (which can be more than one!)
        connections to their siblings are not being computed! this will become part of a different function
        'close_boundary' which is not yet implemented TODO
        """

        k = len(self)-self.layerbounds[-1] # error can not be closed if layerbounds too short
        for kk in range(k):
            self._nbrs.append([])


        for i in range(self.layerbounds[-2], self.layerbounds[-1]):
            for j in self._nbrs[i]:
                self._nbrs[j].append(i)

    
        for kk in range(self.layerbounds[-3],k):
            self._nbrs[kk] = list(np.unique(np.array(self._nbrs[kk])))

        self.closed = True  


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
            for _ in range(self.n-1):
                self.add_layer(self.not_origin)

        elif self.center == "vertex":
            self._create_first_layer()
            self._prepare_duplicate_container()
            for _ in range(self.n-1):
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


# ------------- Neighbours -------------


    def get_nbrs_list(self):
        """
        return neighbour list of entire lattice
        """
        return self._nbrs


    def get_nbrs(self, i):
        """
        return neighbours of cell i as list
        """
        return self._nbrs[i]


        