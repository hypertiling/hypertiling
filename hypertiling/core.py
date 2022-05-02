import numpy as np
import math
import copy
import numba

# relative imports
from .hyperpolygon import HyperPolygon
from .transformation import p2w, moeb_rotate_trafo
from .util import fund_radius

# the main object of this library
# essentially represents a list of polygons which build the hyperbolic lattice
class HyperbolicTiling:
    def __init__(self, p, q, nlayers):
        self.p = p  # number of edges (and thus number of vertices) per polygon
        self.q = q  # number of polygons that meet at each vertex
        self.nlayers = nlayers  # layers of the tessellation

        self.phi = 2*np.pi/self.p  # angle of rotation that leaves the lattice invariant
        self.degphi = 360/self.p
        self.dgts = 8   # rounding digits, default: 8 (do not change, unless you know what you are doing!)
        self.degtol = 1 # sector boundary tolerance during lattice construction

#        self.centerlist = []  # used to keep track of which polygons has already been drawn
        self.fund_poly = self.create_fundamental_polygon()  # central polygon of the tessellation
#        self.lpolygons = [[] for _ in range(self.nlayers)]  # for each layer there is one subarray
        self.polygons = []  # duplicate-free array of polygons of the layer

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

    # constructs the vertices of the fundamental hyperbolic {p,q} polygon
    def create_fundamental_polygon(self):
        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p, self.q)

        for i in range(self.p):
            z = complex(r * math.cos(i*self.phi), r * math.sin(i*self.phi))  # = r*exp(i*phi)
            polygon.verticesP[i] = z
            polygon.verticesW[:, i] = p2w(z)
        return polygon



    # generates the whole lattice by first constructing one 1/p sector, 
    # then uses symmetry to construct the other p-1 sectors

    # in order to avoid problems associated to rounding we construct the
    # fundamental sector a little bit wider than 360/p degrees in filter
    # out rotational duplicates after all layers have been constructed

    def generate(self):
        # prepare list to store polygons 
        self.polygons.append(self.fund_poly)
        # prepare sets which will contain the center coordinates
        # this is used for uniqueness checks later
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
                    for rot_ind in range(1, self.q):

                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon
                        adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)

                        # compute center and angle
                        center = np.round(adj_pgon.centerP(), self.dgts)
                        adj_pgon.find_angle()

                        # cut away cells outside the fundamental sector
                        # allow some tolerance at the upper boundary
                        if 0 <= adj_pgon.angle < self.degphi+self.degtol:

                            # try adding to centerlist; it is a set() and takes care of duplicates
                            lenA = len(centerset)
                            centerset.add(center) 
                            lenB = len(centerset)

                            # this tells us whether an element has actually been added
                            if lenB>lenA:
                                adj_pgon.layer = l+1
                                # add corresponding poly to large list
                                self.polygons.append(adj_pgon)

                            # if angle is in slice, add to centerset_extra
                            if adj_pgon.angle < self.degtol:
                                centerset_extra.add(center)
            startpgon = endpgon
            endpgon = len(self.polygons)

        # free mem of centerset
        del centerset

        # filter out rotational duplicates
        deletelist = []
        for kk, pgon in enumerate(self.polygons):
            if pgon.angle > self.degphi-self.degtol:

                center = moeb_rotate_trafo(pgon.centerP(), -self.phi)
                center = np.round(center, self.dgts) # better use simple distance?

                if center in centerset_extra:
                    deletelist.append(kk)

        self.polygons = list(np.delete(self.polygons, deletelist))


        # fill entire disk by rotating the slice
#        self.angular_replicate(copy.deepcopy(self.polygons), self.p)



    # finds the next polygon by k-fold rotation of polygon around the vertex number ind
    def generate_adj_poly(self, polygon, ind, k):
        z0 =  complex(polygon.verticesP[ind])
        dz0 = polygon.verticesdP[ind]
        polygon.moeb_origin(z0, dz0)  # map vertex at z0 to origin at (0,0)
        polygon.moeb_rotate(k*2*np.pi/self.q)  # rotate the whole polygon k times by 2*pi/q
        polygon.moeb_inverse(z0, dz0)  # map polygon back to former location
        return polygon


    # tessellates the disk by applying a rotation of 2pi/p to the pizza slice
    def angular_replicate(self, polygons, k):
        polygons.pop(0)  # first pgon (partially) lies in every sector and thus need not be replicated
        for p in range(1, k):
            for polygon in polygons:
                pgon = copy.deepcopy(polygon)
                pgon.moeb_rotate(-p*self.phi) # 3s
                pgon.find_angle()
                pgon.find_sector()
                self.polygons.append(pgon)

        # assign each polygon a unique number
        for num, poly in enumerate(self.polygons):
            poly.idx = num + 1


    # populate the "edges" list of all polygons in the tiling
    def populate_edge_list(self, digits=12):
        # note: same neighbour search methods employ the fact that adjacent polygons share an edge
        # hence these will later be identified via floating point comparison and we need to round
        # note: this procedure fails for Weierstrass coordinates, as these
        # are not unique, meaning that the same coordinate can have different representations
        for poly in self.polygons:
            poly.edges = []
            verts = np.round(poly.verticesP[0:-1], digits)
        
            # append edges as tuples
            for i, vert in enumerate(verts[:-1]):
                poly.edges.append((verts[i], verts[i+1]))
            poly.edges.append((verts[-1], verts[0]))



# After the algorithm by D. Dunham (1982)
# works for every valid combination {p,q}
# however produces a lot of duplicates
class HyperbolicTilingDunham:
    def __init__(self, p, q, nlayers):
        self.p = p
        self.q = q
        self.nlayers = nlayers

        # reflection and rotation matrices
        self.b = np.arccosh(np.cos(np.pi / q) / np.sin(np.pi / p))

        self.ReflectPgonEdge = np.array([[-np.cosh(2 * self.b), 0, np.sinh(2 * self.b)],
                                         [0, 1, 0],
                                         [-np.sinh(2 * self.b), 0, np.cosh(2 * self.b)]])
        self.ReflectEdgeBisector = np.array([[1, 0, 0],
                                             [0, -1, 0],
                                             [0, 0, 1]])
        self.ReflectHypotenuse = np.array([[np.cos(2 * np.pi / p), np.sin(2 * np.pi / p), 0],
                                           [np.sin(2 * np.pi / p), -np.cos(2 * np.pi / p), 0],
                                           [0, 0, 1]])

        self.RotP = self.ReflectHypotenuse @ self.ReflectEdgeBisector
        self.RotQ = self.ReflectPgonEdge @ self.ReflectHypotenuse
        self.Rot2P = self.RotP @ self.RotP  # actually boosts the performance
        self.Rot3P = self.Rot2P @ self.RotP
        self.RotCenterG = np.eye(3)  # G for usage in generate()
        self.RotCenterR = np.eye(3)   # R for usage in replicate(...)

        self.fund_poly = self.create_fundamental_polygon()
        self.polygons = [self.fund_poly]

    def create_fundamental_polygon(self):  # constructs the verticesP of the fundamental hyperbolic {p,q} polygon
        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p, self.q)
        angle = np.pi / self.p
        for i in range(self.p):  # for every corner of the polygon
            z = complex(r * np.cos(angle + 2 * np.pi * i / self.p), r * np.sin(angle + 2 * np.pi * i / self.p))
            polygon.verticesP[i] = z
            polygon.verticesW[:, i] = p2w(z)
        return polygon

    def generate(self):
        if self.nlayers == 1:
            return

        for _ in range(self.p):
            RotVertex = self.RotCenterG @ self.RotQ
            self.replicate(self.polygons, RotVertex, self.nlayers - 2, "Edge")
            for _ in range(self.q - 3):
                RotVertex = RotVertex @ self.RotQ
                self.replicate(self.polygons, RotVertex, self.nlayers - 2, "Vertex")

            self.RotCenterG = self.RotCenterG @ self.RotP

    def replicate(self, Polygons, InitialTran, LayersToDo, AdjacencyType):
        poly = copy.deepcopy(self.fund_poly)
        poly.transform(InitialTran)
        Polygons.append(poly)  # appending anything and removing duplicates afterwards is faster
        ExposedEdges = 0
        VertexPgons = 0

        if LayersToDo > 0:
            if AdjacencyType == "Edge":
                ExposedEdges = self.p - 3
                self.RotCenterR = InitialTran @ self.Rot3P
            if AdjacencyType == "Vertex":
                ExposedEdges = self.p - 2
                self.RotCenterR = InitialTran @ self.Rot2P

            for j in range(ExposedEdges):
                RotVertex = self.RotCenterR @ self.RotQ
                self.replicate(Polygons, RotVertex, LayersToDo - 1, "Edge")
                if j < ExposedEdges:  # I do not understand where -3 and -4 comes from
                    VertexPgons = self.q - 1  # was -3, corrected by trial and error
                elif j == ExposedEdges:
                    VertexPgons = self.q - 2  # and -4 in Dunhams paper

                for _ in range(VertexPgons):
                    RotVertex = RotVertex @ self.RotQ
                    self.replicate(Polygons, RotVertex, LayersToDo - 1, "Vertex")

                self.RotCenterR = self.RotCenterR @ self.RotP
