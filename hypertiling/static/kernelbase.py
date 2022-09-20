import numpy as np
import math
import copy
# relative imports
from .hyperpolygon import HyperPolygon
from ..arraytransformation import mfull, mrotate, morigin
from ..transformation import p2w, moeb_rotate_trafo, mymoeb
from ..util import fund_radius

# Magic number: real irrational number \Gamma(\frac{1}{4})
# used as an angular offset, rotates the entire construction by a bit during construction
MANGLE = 3.6256099082219083119306851558676720029951676828800654674333779995


# the main object of this library
# essentially represents a list of polygons which constitute the hyperbolic lattice
class HyperbolicTilingBase:
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

    def __init__(self, p, q, nlayers, center="cell"):

        # main attributes
        self.p = p                  # number of edges (and thus number of vertices) per polygon
        self.q = q                  # number of polygons that meet at each vertex
        self.nlayers = nlayers      # layers of the tessellation
        self.center = center        # tiling can be centered around a "cell" (default) or a "vertex"

        # symmetry angles
        self.phi = 2*math.pi/self.p  # angle of rotation that leaves the lattice invariant when cell centered
        self.qhi = 2*math.pi/self.q  # angle of rotation that leaves the lattice invariant when vertex centered
        self.degphi = 360/self.p   # self.phi in degrees
        self.degqhi = 360/self.q   # self.qhi in degrees

        # technical parameters 
        # do not change, unless you know what you are doing!)
        self.degtol = 1  # sector boundary tolerance

        # # fundamental polygon of the tiling
        # self.fund_poly = self.create_fundamental_polygon(center)

        # prepare list to store polygons 
        self.polygons = []

        if center not in ['cell', 'vertex']:
            raise ValueError('Invalid value for argument "center"!')

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
        return self.polygons[index].verticesP[:self.p]

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
        :return: np.complex128 = center of the polygon
        """
        return self.polygons[index].angle

    def create_fundamental_polygon(self, center='cell', rotate_by=MANGLE):
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
            z = complex(math.cos(i*self.phi), math.sin(i*self.phi))  # = exp(i*phi)
            z = z/abs(z)
            z = r * z
            polygon.verticesP[i] = z

        # if centered around a vertex, shift one vertex to origin
        if center == 'vertex':
            morigin(self.p, complex(r, 0), polygon.verticesP)
            vertangle = math.atan2(polygon.verticesP[1].imag, polygon.verticesP[1].real)
            mrotate(self.p, vertangle, polygon.verticesP)
            polygon.angle = math.degrees(math.atan2(polygon.verticesP[self.p].imag, polygon.verticesP[self.p].real))
            polygon.angle += 360 if polygon.angle < 0 else 0

        mrotate(self.p, -2*math.pi/360*rotate_by, polygon.verticesP)

        return polygon


class KernelCommon(HyperbolicTilingBase):
    """
    Commonalities
    """

    def __init__(self, p, q, n, center):
        super(KernelCommon, self).__init__(p, q, n, center)

    def replicate(self):
        """
        tessellate the entire disk by replicating the fundamental sector
        """
        if self.center == 'cell':
            self.angular_replicate(copy.deepcopy(self.polygons), self.p)
        elif self.center == 'vertex':
            self.angular_replicate(copy.deepcopy(self.polygons), self.q)

    def generate(self):
        """
        do full construction
        """
        self.generate_sector()
        self.replicate()

    def generate_adj_poly(self, polygon, ind, k):
        """
        finds the next polygon by k-fold rotation of polygon around the vertex number ind
        """
        mfull(self.p, k*self.qhi, ind, polygon.verticesP)
        return polygon

    # tessellates the disk by applying a rotation of 2pi/p to the pizza slice
    def angular_replicate(self, polygons, k):
        if self.center == 'cell':
            polygons.pop(0)  # first pgon (partially) lies in every sector and thus need not be replicated
            angle = self.phi
            k = self.p
        elif self.center == 'vertex':
            angle = self.qhi
            k = self.q

        for p in range(1, k):
            for polygon in polygons:
                pgon = copy.deepcopy(polygon)
                mrotate(self.p, -p*angle, pgon.verticesP)
                pgon.angle = math.degrees(math.atan2(pgon.verticesP[self.p].imag, pgon.verticesP[self.p].real))
                pgon.angle += 360 if pgon.angle < 0 else 0
                pgon.sector = math.floor(pgon.angle/(360/k))
                self.polygons.append(pgon)

        # assign each polygon a unique number
        for num, poly in enumerate(self.polygons):
            poly.idx = num + 1

    # populate the "edges" list of all polygons in the tiling
    # untested!!
    def populate_edge_list(self, digits=12):
        # note: same neighbour search methods employ the fact that adjacent polygons share an edge
        # hence these will later be identified via floating point comparison and we need to round
        for poly in self.polygons:
            poly.edges = []
            verts = round(poly.verticesP[0:-1], digits)
        
            # append edges as tuples
            for i, vert in enumerate(verts[:-1]):
                poly.edges.append((verts[i], verts[i+1]))
            poly.edges.append((verts[-1], verts[0]))

    def rotate(self, angle, deg=False):
        """
        Rotates the whole tiling around the origin.
        
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
            morigin(self.p, -z, poly.verticesP)

    def add_layer(self):
        """ constructs an additional layer for a given tiling object. Not implemented for GRK and Legacy Kernel yet. """

        if kernel not in ['GRK', 'Dunham']:
            print('This function has not been implemented for this kernel yet. We are working on this!')
            return

        newpolygons = []

        if kernel == 'SFK':  # former 'kernelmanu'
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

        elif kernel == 'SPK':  # former 'kernelflo'
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
