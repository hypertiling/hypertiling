import numpy as np
import math
import copy
# relative imports
from .hyperpolygon import HyperPolygon
from ..arraytransformation import mfull, mrotate, morigin
from ..transformation import p2w, moeb_rotate_trafo, mymoeb
from ..util import fund_radius

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
        self.degtol = 1 # sector boundary tolerance
        
        # angular offset, rotates the entire construction by a bit during construction
        if center == "cell":
            self.mangle = self.degphi/math.sqrt(5) 
        elif center == "vertex":
            self.mangle = self.degqhi/math.sqrt(5)


        # fundamental polygon of the tiling
        self.fund_poly = self.create_fundamental_polygon(center)

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


    def create_fundamental_polygon(self, center='cell'):
        """
        Constructs the vertices of the fundamental hyperbolic {p,q} polygon

        Parameters
        ----------

        center : str
            decides whether the fundamental cell is construct centered at the origin ("cell", default) 
            or with the origin being one of its vertices ("vertex")


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

        mrotate(self.p, -2*math.pi/360*self.mangle, polygon.verticesP)

        return polygon

class KernelCommon(HyperbolicTilingBase):
    """
    Commonalities
    """

    def __init__ (self, p, q, n, center):
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

