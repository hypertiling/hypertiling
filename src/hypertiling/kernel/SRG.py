import numpy as np
import math
import copy
from ..util import n_cell_centered, n_vertex_centered
from ..ion import htprint
from ..arraytransformation import mfull, morigin, multi_rotation_around_vertex
from .SRG_util import DuplicateContainerCircular
from .SR_base import KernelRotationalCommon

PI2 = 2 * np.pi


class StaticRotationalGraph(KernelRotationalCommon):
    '''
    Static Rotational Graph (SRG) kernel

    The default kernel of the hypertiling package

    It provides great flexibility by allowing to construct and dynamically manipulate 
    hyperbolic tilings; unlike the other static rotational kernels, here the neighbours 
    are computed during construction of the tiling
    '''

    def __init__ (self, p, q, n, **kwargs):
        super(StaticRotationalGraph, self).__init__(p, q, n, **kwargs)

        # a place to collect neighbour information
        self.nbrs = {}

        # a place to collect the cells which constitute the tiling
        self.polygons = {}

        # a place to store indices of cells which are "exposed" 
        # (they have incomplete neighbourhood information)
        self.exposed = []

        # helpers
        self.globcount = 0
        self.layercount = 0

        # construct tiling
        self.generate()

    def __iter__(self):
        for poly in self.polygons.values():
            # (center, vertex_1, vertex_2, ..., vertex_p)
            yield np.roll(poly.verticesP,1)

    def __len__(self):
        return len(self.polygons)


    def remove_cells(self, deletelist):
        '''
        deletelist : List[int]
            list of polygon indices to be removed from the tiling;
            note that an error is thrown if an index can not be found in the lattice
        '''

        for idx in deletelist:
            # remove from neighbour list
            # remove occurence in neighour lists of other cells
            try:
                nbrs_of_idx = self.nbrs[idx]
                for nb in nbrs_of_idx:
                    try:
                        self.nbrs[nb].remove(idx)
                    except ValueError:
                        pass
                del self.nbrs[idx]
            except KeyError:
                pass

            # remove from exposed cells
            try:
                self.exposed.remove(idx)
            except ValueError:
                pass

            # remove from duplicate container
            try:
                z = self.polygons[idx].centerP()
                self.dplcts.remove_by_idx(z, idx)
            except:
                pass

            # remove from polygon list
            try:
                del self.polygons[idx]
            except KeyError:
                pass
            



    def add_layer(self, addlist=None, filter=None):
        """
        Create new cells in an existing tiling

        deletelist : List[int]
            list of polygon indices, all adjacent spots around those cells are filled with new cells
            In case no input is provided, the list of currently "exposed" cells is used
        filter : callable
            user-defined filter function which allows to limit the construction to certain
            spatial regions based on the (center) coordinate of the cells
        """

        if addlist is None:
            polylist = self.exposed
        else:
            polylist = addlist

        if filter is None:
            filter = self.filter_always_pass

        newexposed = []

        for pgonidx in polylist:
            pgon = self.polygons[pgonidx]

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

                    # check whether candidate polygon is _not_ close to the origin
                    if filter(center):   

                        # check whether candidate polygon already exists
                        duplicate, idx = self.dplcts.is_duplicate(center)
                        if not duplicate:

                            # create copy
                            polycopy = copy.deepcopy(pgon)

                            # generate adjacent polygon
                            adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)
                            adj_pgon.layer = self.layercount
                            # add to tiling
                            newpolyidx = self._add_pgon(adj_pgon)
                            collect_nbrs.append(newpolyidx)
                            newexposed.append(newpolyidx)

                        else:
                            collect_nbrs.append(idx)
                    else:
                        collect_nbrs.append(0)

            collect_nbrs = np.array(collect_nbrs)
            collect_nbrs = collect_nbrs[collect_nbrs != self.counter]

            nbr_list = list(np.unique(collect_nbrs))
            self.nbrs[pgon.idx] = nbr_list
            
            # establish mutual connections
            # i.e.connect new cells to their parent polygons
            # important note: child will only be connected to those parents from which they have been
            # generated (see documentation notebook)
            for nb in nbr_list:
                self.nbrs[nb].append(pgon.idx)
                self.nbrs[nb] = list(set(self.nbrs[nb]))

            self.counter += 1
            self.layercount += 1

        # we have constructed neighbours around exposed cells, hence
        # they are no longer exposed; but the newly created ones are
        if addlist is None:
            self.exposed = newexposed
        # merge lists of existing exposed cells which have not been considered
        # and new exposed cells
        else:
            self.exposed = [x for x in self.exposed if (x not in addlist)]
            self.exposed += newexposed


    def _add_pgon(self, pgon):
        """
        Add new polygon to tiling
        """

        # assign index to new polygon
        pgon.idx = self.globcount
        # increment global count
        self.globcount += 1
        # add polygon to container
        self.polygons[pgon.idx] = pgon
        # add empty list for this poly in nbrs
        self.nbrs[pgon.idx] = []
        # add to duplicate container
        self.dplcts.add(pgon.centerP(), pgon.idx)

        # return index of new polygons
        return pgon.idx


    def _prepare_duplicate_container(self):
        """
        prepare containers for duplicate checks
        init with origin, set angle artificially to phi/2
        """

        self.dplcts = DuplicateContainerCircular(self.p * self.q)       


    def _create_first_layer(self, rotate_by=None):
        """
        generate the first layer
        this is one polygon for cell-centered and q polygons for vertex-centered
        """

        # create fundamental polygon
        self.fund_poly = self.create_fundamental_polygon(rotate_by)

        # prepare polygon counter
        self.counter = 0

        # tiling centered around cell
        # add fundamental cell and set bounds of current layer
        if self.center == "cell":
            self.fund_poly.moeb_origin(0.000001) 
            # necessary for technical reasons since the duplicate container is singular at the origin
            # TODO: add warning

            self._add_pgon(self.fund_poly)
            self.exposed = [0]


        # tiling centered around vertex
        if self.center == 'vertex':

            # shift fundamental polygon such that one of its vertices is on the origin
            vertidx = 0           
            morigin(self.p, self.fund_poly.verticesP[vertidx], self.fund_poly.verticesP)
            
            # generate the q polygons of the first layer
            for rot_ind in range(self.q):
                polycopy = copy.deepcopy(self.fund_poly)
                adj_pgon = self._generate_adj_poly(polycopy, vertidx, rot_ind)
                newpgonidx = self._add_pgon(adj_pgon)
                self.exposed.append(newpgonidx)


    def generate(self):
        """
        construct full tiling by calling the add_layer function repeatedly
        """
        self.polygons = {}
        self._prepare_duplicate_container()
        self._create_first_layer()

        for i in range(self.n-1):
            self.add_layer()


    def generate_adj_poly(self, polygon, ind, k):
        """
        construct new polygon by k-fold rotation of "polygon" around its vertex "ind"
        """
        mfull(self.p, k * self.qhi, ind, polygon.verticesP)
        return polygon
    

    def _peformance_warning(self):
        if self.center == "vertex":
            n_est = n_vertex_centered(self.p, self.q, self.n)
        if self.center == "center":
            n_est = n_cell_centered(self.p, self.q, self.n)

        if n_est > 1e6:
            htprint("Warning", "You requested a very large tiling and might want to consider using a different construction kernel (compare documentation)!")



# ------------- Filters -------------

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
        return self.nbrs


    def get_nbrs(self, i):
        """
        return neighbours of cell i as list
        """
        return self.nbrs[i]


        