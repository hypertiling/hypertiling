import numpy as np
import math
import copy

# relative imports
from .hyperpolygon import HyperPolygon
from .transformation import p2w, moeb_rotate_trafo
from .util import fund_radius
from .distance import disk_distance


# the main object of this library
# essentially represents a list of polygons which constitute the hyperbolic lattice
class HyperbolicTiling:
    def __init__(self, p, q, nlayers, center='cell'):

        # main attributes
        self.p = p                  # number of edges (and thus number of vertices) per polygon
        self.q = q                  # number of polygons that meet at each vertex
        self.nlayers = nlayers      # layers of the tessellation
        self.center = center        # tiling can be centered around a "cell" (default) or a "vertex"

        # symmetry angles
        self.phi = 2*np.pi/self.p  # angle of rotation that leaves the lattice invariant when cell centered
        self.qhi = 2*np.pi/self.q  # angle of rotation that leaves the lattice invariant when vertex centered
        self.degphi = 360/self.p   # self.phi in degrees
        self.degqhi = 360/self.q   # self.qhi in degrees

        # technical parameters 
        # do not change, unless you know what you are doing!)
        self.dgts = 8   # rounding digits, default: 8
        self.accuracy = 10**(-self.dgts) # numerical accuracy
        self.degtol = 1 # sector boundary tolerance during construction
        self.mangle = self.degphi/2 # angular offset, rotates the entire construction; must not be larger than 360-360/p!!!


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


    # constructs the vertices of the fundamental hyperbolic {p,q} polygon
    def create_fundamental_polygon(self, center='cell'):
        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p)

        for i in range(self.p):
            z = complex(r * math.cos(i*self.phi), r * math.sin(i*self.phi))  # = r*exp(i*phi)
            polygon.verticesP[i] = z
            polygon.verticesW[:, i] = p2w(z)

        # if centered around a vertex, shift one vertex to origin
        if center == 'vertex':
            polygon.moeb_origin(complex(r, 0))
            polygon.find_angle()
            vertangle = np.arctan2(polygon.verticesP[1].imag, polygon.verticesP[1].real)
            polygon.moeb_rotate(vertangle)
            polygon.find_angle()

        polygon.moeb_rotate(-2*np.pi/360*self.mangle)

        return polygon



    # generates the whole lattice by first constructing one 1/p sector, 
    # then uses symmetry to construct the other p-1 sectors

    # in order to avoid problems associated to rounding we construct the
    # fundamental sector a little bit wider than 360/p degrees in filter
    # out rotational duplicates after all layers have been constructed

    def generate(self):

        # add fundamental polygon to list
        self.polygons.append(self.fund_poly)

        # angle width of the fundamental sector
        sect_angle     = self.phi
        sect_angle_deg = self.degphi
        if self.center == "vertex":
            sect_angle     = self.qhi
            sect_angle_deg = self.degqhi

        # prepare sets which will contain the center coordinates
        # will be used for uniqueness checks
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
                    for rot_ind in range(self.q):

                        # create copy
                        polycopy = copy.deepcopy(pgon)

                        # generate adjacent polygon
                        adj_pgon = self.generate_adj_poly(polycopy, vert_ind, rot_ind)


                        # compute center and angle
                        center = np.round(adj_pgon.centerP(), self.dgts)
                        adj_pgon.find_angle()

                        # cut away cells outside the fundamental sector
                        # allow some tolerance at the upper boundary
                        if self.mangle <= adj_pgon.angle < sect_angle_deg+self.degtol+self.mangle:

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
                            if self.mangle < adj_pgon.angle < self.degtol+self.mangle:
                                centerset_extra.add(center)
            startpgon = endpgon
            endpgon = len(self.polygons)



            if self.numerically_unstable_upper(l):
                print("Numerical accuracy exhausted; no more layers will be constructed; automatic shutdown")
                break

            if self.numerically_unstable_lower(l):
                print("Accumulated numerical errors have become too large; no more layers will be constructed; automatic shutdown")
                break



        # flatten the list
        for curr_layer in self.lpolygons:
            for polygon in curr_layer:
                self.polygons.append(polygon)

        # free mem of centerset
        del centerset

        # filter out rotational duplicates
        deletelist = []
        for kk, pgon in enumerate(self.polygons):
            if pgon.angle > sect_angle_deg-self.degtol+self.mangle:

                center = moeb_rotate_trafo(pgon.centerP(), -sect_angle)

                center = np.round(center, self.dgts) # better use simple distance?

                if center in centerset_extra:
                    deletelist.append(kk)

        self.polygons = list(np.delete(self.polygons, deletelist))


        # fill entire disk by rotating the slice
        if self.center == 'cell':
            self.angular_replicate(copy.deepcopy(self.polygons), self.p)
        elif self.center == 'vertex':
            self.angular_replicate(copy.deepcopy(self.polygons), self.q)

    # check whether the true "embedding" distance between cells in layer l comes close
    # to the rounding accuracy
    def numerically_unstable_upper(self, l, tolfactor=10, samplesize=10):

        # randomly pick a number of sites from l-th layer
        layersize = len(self.lpolygons[l])
        true_dists = []
        for i in range(samplesize):
            rndidx = np.random.randint(layersize)

            # generate an adjacent cell
            mother = self.lpolygons[l][rndidx]
            child  = self.generate_adj_poly(copy.deepcopy(mother), 0, 1)

            # compute the true (non-geodesic) distance
            true_dist = np.abs(mother.centerP-child.centerP)
            true_dists.append(true_dist)

        # if this distances comes close to the rounding accuracy
        # two cells can no longer be reliably distinguished
        if np.min(true_dist) < self.accuracy*tolfactor:
            return True
        else:
            return False


    # we know which geodesic distance two adjancent cells are supposed to have;
    # here we take a sample of cells from the l-th layer and compute mutual 
    # distances; if one of those is significantly off compared to the expected
    # value we are about to enter a dangerous regime in terms of rounding errors
    def numerically_unstable_lower(self, l, tolfactor=10, samplesize=100):

        # innermost layers are always fine, do nothing
        if l<3:
            return False

        # take a sample of cells and compute their distances
        disk_distances = []
        for j1, pgon1 in enumerate(self.lpolygons[l][0:samplesize]):
            for j2, pgon2 in enumerate(self.lpolygons[l][0:samplesize]):
                if j1 != j2:
                    disk_distances.append(disk_distance(pgon1.centerP, pgon2.centerP))

        # we are interested in the minimal distance (can be interpreted as an 
        # upper bound on the accumulated error)
        mindist = np.min(np.array(disk_distances))

        # the reference distance
        refdist = disk_distance(self.fund_poly.centerP, self.lpolygons[1][0].centerP)

        # if out arithmetics worked error-free, mindist = refdist
        # in practice, it does not, so we compute the difference
        # if it comes close to the rounding accuracy, adjacency can no longer
        # by reliably resolved and we are about to enter a possibly unstable regime
        if np.abs(mindist-refdist) > self.accuracy/tolfactor:
            return True
        else:
            return False



    # finds the next polygon by k-fold rotation of polygon around the vertex number ind
    def generate_adj_poly(self, polygon, ind, k):
        polygon.tf_full(ind, k*self.qhi)
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
                pgon.moeb_rotate(-p*angle)
                pgon.find_angle()
                pgon.find_sector(k)
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
        polygon = HyperPolygon(self.p)
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
