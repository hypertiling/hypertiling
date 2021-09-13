import numpy as np
import copy

# relative imports
from .hyperpolygon import HyperPolygon
from .transformation import *
from .util import fund_radius, remove_duplicates, find_num_of_pgons_73
from .distance import disk_distance
from .old import remove_overflow

# Tessellation algorithm written by F. Dusel
# uses Moebius transformations to compute adjacent polygons by effectively mirroring the existent polygon at its edges
# works perfectly for {7,3}, but is not stable for non-{7,3} tessellations
class HyperbolicTiling73Experimental:
    def __init__(self, p, q, nlayers):
        self.p = p
        self.q = q

        self.s = fund_radius(self.p, self.q) / 2  # np.sin(2*np.pi / self.p) * fund_radius(self.p,self.q)
        self.phi = 2*np.pi/self.p  # acute angle of a 1/p slice of the disk
        self.dgts = 10  # decimal place that is rounded -> precision

        self.cutoff = 1  # tiling condition that tessellations end after cutoff; no cutoff for ==1
        avg_d = 2*self.s*(1/2+np.sin(self.phi/2))
        self.nlayers = nlayers if self.cutoff == 1 else int(2 + np.floor(disk_distance(0, self.cutoff)/avg_d))
        self.nsectors = 360

        self.fund_poly = self.create_fundamental_polygon()  # central polygon
        self.lpolygons = [[] for _ in range(self.nlayers)]  # for each layer there is one subarray
        self.polygons = []
        self.centerlist = []  # used to keep track of which polygons has already been drawn

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

    def create_fundamental_polygon(self):  # constructs the vertices of the fundamental hyperbolic {p,q} polygon
        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p, self.q)
        polygon.number = 1
        polygon.layer = 1

        for i in range(self.p):
            z = complex(r * np.cos(i*self.phi), r * np.sin(2 * np.pi * i / self.p))
            polygon.verticesP[i] = z
            polygon.verticesW[:, i] = p2w(z)
        return polygon

    def generate(self):
        self.lpolygons[0].append(self.fund_poly)
        self.centerlist.append(np.round(self.fund_poly.centerP, self.dgts))
        sectors = np.arange(0, self.nsectors+1)
        for l in range(1, self.nlayers):
            for pgon in self.lpolygons[l-1]:  # finds all neighbors of layer l, only adds those that...
                for vert_ind in range(self.p):  # finds the neighbor of each edge of pgon
                    polycopy = copy.deepcopy(pgon)
                    adj_pgon = self.generate_adj_poly(polycopy, vert_ind)  # mirror pgon at edge
                    center = np.round(adj_pgon.centerP, self.dgts)
                    if center not in self.centerlist and abs(center) <= self.cutoff:  # ...are no duplicates
                        adj_pgon.find_angle(360)
                        if adj_pgon.sector in sectors:  # and in the same sector
                            adj_pgon.layer = l+1
                            self.centerlist.append(center)
                            self.lpolygons[l].append(adj_pgon)

        pnum = 1
        for lst in self.lpolygons:  # flattening the list, only including the right sector polygons
            for polygon in lst:
                if polygon.sector in sectors[:-1]:
                    polygon.number = pnum  # assigning each polygon a unique number for later purposes
                    pnum += 1
                    self.polygons.append(polygon)

        # self.find_boundary_width()  # used for finding the radial distance the outmost layer covers
        self.angular_replicate(copy.deepcopy(self.polygons))  # uses symmetry to fill the plane by rotating the slice

    def generate_adj_poly(self, polygon, vert_ind):
        z0 = polygon.verticesP[vert_ind]
        polygon.moeb_origin(z0)  # map vertex to origin
        phi = np.arctan2(polygon.verticesP[vert_ind-1].imag, polygon.verticesP[vert_ind-1].real)
        phi += 2 * np.pi if phi < 0 else 0  # such that phi in [0, 2pi]
        polygon.moeb_rotate(phi)  # rotate such that edge is purely real
        polygon.moeb_translate(self.s)  # translate such that the edge is symmetric to the im-axis
        polygon.mirror()  # reflect at edge
        polygon.moeb_inverse(z0, phi, self.s)  # reverses all previous transformations
        return polygon

    def angular_replicate(self, polygons):  # tessellates the disk by applying a rotation of 2pi/p to the pizza slice
        polygons.pop(0)  # first pgon is in each 1/p sector
        for p in range(1, self.p):
            for ind, polygon in enumerate(polygons):
                pgon = copy.deepcopy(polygon)
                pgon.rotate(-p*self.phi)  # - stems from definition of moeb_rotate()
                pgon.find_angle(360)
                self.polygons.append(pgon)

        for num, poly in enumerate(self.polygons):
            poly.number = num+1

    # the following are experimental, untested, and currently not being used

    def generate_alt(self):
        p_ind = 0
        pcounter = 2
        self.draw_neighbors(self.fund_poly, self.nlayers-1)
        print(len(self.polygons))
        print("soll = ", (find_num_of_pgons_73(self.nlayers) - 1) / 7 + 1)

        lim = 1+(find_num_of_pgons_73(self.nlayers)-1)/7
        while pcounter < 1+(find_num_of_pgons_73(self.nlayers)-1)/7:

            for p_ind in range(0, int(lim)):
                # while pcounter < find_num_of_pgons73(self.nlayers):
                for vert_ind in range(self.p):
                    polycopy = copy.deepcopy(self.polygons[p_ind])
                    polycopy = self.generate_adj_poly(polycopy, vert_ind)
                    center = np.round(polycopy.centerP, self.dgts)
                    if center not in self.centerlist:
                        polycopy.find_angle(360)
                        if polycopy.sector == 0:
                            self.polygons.append(polycopy)
                            self.centerlist.append(center)
                            polycopy.number = pcounter
                            pcounter += 1
            p_ind += 1

        self.polygons = remove_duplicates(self.polygons)
        print(len(self.polygons))
        pgons_to_remove = (find_num_of_pgons_73(self.nlayers)-1)/7+1
        print("to remove: ", len(self.polygons) - pgons_to_remove)
        self.polygons = remove_overflow(self.polygons, num=len(self.polygons) - pgons_to_remove)

        self.polygons.pop(0)  # exclude first element to avoid duplicates
        polygons = copy.deepcopy(self.polygons)
        for p in range(0):
            self.angular_replicate(polygons)
        # self.polygons = remove_duplicates(self.polygons)
        self.polygons.insert(0, self.fund_poly)  # reinsert first polygon

    def generate_recursive(self):
        if self.nlayers == 1:
            return
        self.draw_neighbors(self.fund_poly, self.nlayers - 1)

    def draw_neighbors(self, polygon, ltd):  # finish n of 1 poly first, then move on
        polygon.number = ltd
        self.polygons.append(polygon)
        self.centerlist.append(np.round(polygon.centerP, self.dgts))

        if ltd > 0:  # inefficient
            neighbors = []
            for v in range(self.p):
                pgon = copy.deepcopy(polygon)
                poly = self.generate_adj_poly(pgon, v)
                # print(abs(polygon.centerP))
                center = np.round(poly.centerP, self.dgts)
                if center not in self.centerlist:
                    poly.find_angle(360)
                    if poly.sector == 0:
                        poly.number = ltd-1
                        self.polygons.append(poly)
                        self.centerlist.append(center)
                        # neighbors.append(poly)
                        self.draw_neighbors(poly, ltd-1)

            # for _ in neighbors:
            #     self.draw_neighbors(_, ltd-1)

    def find_boundary_width(self):
        rmax, rmin = 0, disk_distance(0, self.polygons[1].centerP)
        for poly in self.polygons:
            if poly.layer == self.nlayers:
                d = disk_distance(0, poly.centerP)
                rmax = d if d>rmax else rmax
                rmin = d if d<rmin else rmin

        print(rmax-rmin, (rmax-rmin)/rmin)