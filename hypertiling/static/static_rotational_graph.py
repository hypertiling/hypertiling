import numpy as np
import math
import copy
from .hyperpolygon import HyperPolygon

from ..arraytransformation import mfull, mrotate, morigin, multi_rotation_around_vertex
from ..util import fund_radius, euclidean_center
from ..geodesics import geodesic_midpoint
from ..ion import htprint
from ..arraytransformation import multi_rotation_around_vertex
from .static_rotational_graph_util import DuplicateContainerCircular

PI2 = 2 * np.pi

# Magic number: transcendental number (Champernowne constant)
# used as an angular offset, rotates the entire construction by a bit during construction
MAGICANGLE = np.radians(0.1234567891011121314151617181920212223242526272829303132333)


# the main object of this library
# essentially represents a list of polygons which constitute the hyperbolic lattice
class KernelStaticRotationalGraph:
    """
    Base class of a hyperbolic tiling object

    Attributes
    ----------


    Methods
    -------
    __getitem__(idx)
        returns the idx-th HyperPolygon in the tiling

    __iter__()
        traverses throught all HyperPolygons in the tiling

    __next__()
        returns the next HyperPolygon in the tiling

    __len__()
        returns the size of the tiling, which is the number of cells

    """

    def __init__(self, p, q, nlayers, center="cell", autogenerate=True, radius=None):

        # main attributes
        self.p = p  # number of edges (and thus number of vertices) per polygon
        self.q = q  # number of polygons that meet at each vertex
        self.nlayers = nlayers  # layers of the tessellation
        self.center = center  # tiling can be centered around a "cell" (default) or a "vertex"
        self.radius = radius # a cut-off radius (implement me!)
        self.autogenerate = autogenerate # determines whether the lattice is constructed upon class instantiation or only after call to self.generate

        # half fundamental radius
        self.fr2 = fund_radius(self.p, self.q) / 2

        # symmetry angles
        self.phi = 2 * math.pi / self.p  # angle of rotation that leaves the lattice invariant when cell centered
        self.qhi = 2 * math.pi / self.q  # angle of rotation that leaves the lattice invariant when vertex centered

        # prepare list to store polygons 
        self.polygons = []
        # prepare list to store neighours
        self.nbrs = []

        if center not in ['cell', 'vertex']:
            raise ValueError('[hypertiling] Error: Invalid value for argument "center"!')

        # construct tiling
        if self.autogenerate:
            self.generate()


    def __getitem__(self, idx):
        return self.polygons[idx]

    def __iter__(self):
        self.iterctr = 0
        self.itervar = self.polygons[self.iterctr]
        return self

    def __next__(self):
        if self.iterctr < len(self.polygons):
            retval = self.polygons[self.iterctr]
            self.iterctr += 1
            return retval
        else:
            raise StopIteration

    def __len__(self):
        return len(self.polygons)

    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.array[np.complex128][p] = vertices of the polygon
        """
        return self.polygons[index].verticesP[:-1]

    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.complex128 = center of the polygon
        """
        return self.polygons[index].verticesP[-1]

    def get_sector(self, index: int) -> int:
        """
        Returns the sector, the polygon at index refers to.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: int = number of the sector
        """
        return self.polygons[index].sector

    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: float = angle of the polygon
        """
        return self.polygons[index].angle

    def get_layer(self, index: int) -> int:
        """
        Returns the layer to the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: int = layer of the polygon
        """
        return self.polygons[index].layer

    def create_fundamental_polygon(self, rotate_by=MAGICANGLE):
        """
        Constructs the vertices of the fundamental hyperbolic {p,q} polygon

        Parameters
        ----------
        center : str
            decides whether the fundamental cell is construct centered at the origin ("cell", default) 
            or with the origin being one of its vertices ("vertex")
        rotate_by : float
            angle of rotation of the fundamental polygon, default is the magic angle mangle
        """

        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p)

        for i in range(self.p):
            z = complex(math.cos(i * self.phi), math.sin(i * self.phi))  # = exp(i*phi)
            z = z / abs(z)
            z = r * z
            polygon.verticesP[i] = z

        # rotate by angle (to get away from the coordinate axis)
        mrotate(self.p, -rotate_by, polygon.verticesP)

        return polygon


    def generate_first_layer(self):
        """
        generate the first layer
        """


        # create fundamental polygon
        self.fund_poly = self.create_fundamental_polygon()

        # prepare polygon counter
        self.counter = 0

        # tiling centered around cell
        # add fundamental cell and set bounds of current layer
        if self.center == "cell":
            self.polygons.append(self.fund_poly)
            self.outmost_layer_lower = 0
            self.outmost_layer_upper = 1


        # tiling centered around vertex
        if self.center == 'vertex':
            print("nbrs still buggy for vertex centered")

            # shift fundamental polygon such that one of its vertices is on the origin
            vertidx = 0           
            morigin(self.p, self.fund_poly.verticesP[vertidx], self.fund_poly.verticesP)
            
            # generate the q polygons of the first layer
            for rot_ind in range(self.q):
                polycopy = copy.deepcopy(self.fund_poly)
                adj_pgon = self.generate_adj_poly(polycopy, vertidx, rot_ind)
                self.polygons.append(adj_pgon)

            self.outmost_layer_lower = 0
            self.outmost_layer_upper = self.q

        
        # prepare containers for duplicate checks
        # init with origin, set angle artificially to phi/2
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



    def generate(self):
        """
        do full construction
        """


        # clear tiling
        self.polygons = []

        # add layers repeatedly until nlayers is reached
        if self.center == "cell":
            self.generate_first_layer()
            for _ in range(self.nlayers-1):
                self.add_layer(self.not_origin)

        elif self.center == "vertex":
            self.generate_first_layer()
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




# ------------- Refinements -------------


    def refine_lattice(self, iterations=1):
        """ 
        Refine a regular lattice, by subdividing each triangle into four new polygons
        If the tiling is not triangular, in the first step, all cells will be subdivided
        into p triangular cells
        Note that new cells are not isometric anymore!
        
        Parameters
        ----------
        
        iterations: int
            Determines how many times the lattice will be refined; for each iteration the
            total number of polygons will be multiplied by a factor of four
            
        """

        if iterations == 0:
            return

        # if tiling is not triangular, the first refinement steps subdivided all cells
        # into p triangles
        if self.p > 3:
            newpolygons = []
            for pgon in self.polygons:
                for vrtx in range(self.p):
                    child = HyperPolygon(3) 
                    child.verticesP[0] = pgon.verticesP[vrtx]
                    child.verticesP[1] = pgon.verticesP[(vrtx+1)%self.p]
                    child.verticesP[2] = pgon.verticesP[-1]
                    child.verticesP[3] = euclidean_center(child.verticesP[:-1])
                    child.layer = pgon.layer
                    newpolygons.append(child)
            self.polygons = newpolygons
            iterations -= 1 # we have already done one iteration

        
        for _ in range(iterations):
            p = 3 # we use this quite frequently, hence the short form
            newpolygons = []  # stores the new polygons
            for num, pgon in enumerate(self.polygons):  # find the new vertices of each polygon
                ref_vertices = []  # stores newly found vertices through refinement
                # loop through polygon edges
                for vrtx in range(p):
                    # find geodesic midpoint
                    zm = geodesic_midpoint( pgon.verticesP[vrtx], pgon.verticesP[(vrtx+1)%p] )
                    ref_vertices.append(zm)


                # one "mother" triangle bears 4 "children" triangles, one in its mid
                # and three that each share one vertex with their mother

                child = HyperPolygon(p)  # the center triangle whose vertices are the newly found refined ones
                child.layer = pgon.layer
                for i in range(p):
                    child.verticesP[i] = ref_vertices[i]
                child.verticesP[-1] = pgon.centerP()  # the center triangle shares its center with its mother
                child.idx = 4*num+1  # assigning a unique number
                newpolygons.append(child)

                for vrtx in range(p):  # for each vertex of the mother triangle that is being refined
                    child = HyperPolygon(p)  # these are the non-center children
                    vP = [pgon.verticesP[vrtx], ref_vertices[vrtx], ref_vertices[vrtx-1]]
                    for i in range(p):
                        child.verticesP[i] = vP[i]
                    child.verticesP[-1] = euclidean_center(child.verticesP[:-1])
                    child.idx = (4*num+1)+1+vrtx  # assign a unique number
                    newpolygons.append(child)

            self.polygons = newpolygons
        return


