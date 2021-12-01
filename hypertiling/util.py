import copy
import numpy as np

from .distance import weierstrass_distance
from .hyperpolygon import HyperPolygon
from .transformation import p2w


# radius of the fundamental (and every other) polygon
def fund_radius(p, q):
    num = np.cos(np.pi / p + np.pi / q)
    denom = np.cos(np.pi / p - np.pi / q)
    return np.sqrt(num / denom)


# returns the polygons of the refined lattice for a given {3, 7} tiling with N polygons
# thus, it returns 4*N polygons
# this can be done faster by once again using symmetry, e.g. with angular_replicate() in core
def refine_lattice(tilingobj, n):  # n is the number of refinements
    if n == 0:  # recursive function terminates for n==0
        return tilingobj  # and returns an instance of the chosen TilingClass

    tiling = copy.deepcopy(tilingobj)  # needed to avoid (I think) pointer issues
    p, q = tiling.p, tiling.q
    ref_lattice = []  # stores the new polygons
    for num, pgon in enumerate(tiling.polygons):  # find the new vertices of each polygon
        ref_vertices = []  # stores newly found vertices through refinement
        for vrtx in range(p):
            x_avg = sum(np.real([pgon.verticesP[vrtx], pgon.verticesP[(vrtx+1)%p]]))/p  # avg x of vert-th edge
            y_avg = sum(np.imag([pgon.verticesP[vrtx], pgon.verticesP[(vrtx+1)%p]]))/p  # avg y ...
            ref_vertices.append(1.5*complex(x_avg, y_avg))  # 1.5 had to be found by trial and error...

        # one "mother" triangle bears 4 "children" triangles, one in its mid
        # and three that each share one vertex with their mother
        child = HyperPolygon(p, q)  # the center triangle whose vertices are the newly found refined ones
        child.verticesP = np.array(ref_vertices)
        child.centerP = pgon.centerP  # the center triangle shares its center with its mother
        child.centerW = p2w(child.centerP)
        child.number = 4*num+1  # assigning a unique number
        ref_lattice.append(child)

        for vrtx in range(p):  # for each vertex of the mother triangle that is being refined
            child = HyperPolygon(p, q)  # these are the non-center children
            vP = [pgon.verticesP[vrtx], ref_vertices[vrtx], ref_vertices[vrtx-1]]
            child.verticesP = np.array(vP)
            center_x = sum(np.real(child.verticesP))/p  # trick: average over the xs and ys of the vertices to get
            center_y = sum(np.imag(child.verticesP))/p  # ... an approximate value for centerP
            child.centerP = complex(center_x, center_y)
            child.centerW = p2w(child.centerP)
            child.number = (4*num+1)+1+vrtx  # unique number
            ref_lattice.append(child)

    # print("right length after refinement:", 4 * len(tiling.polygons) == len(ref_lattice))  # optional check
    tiling.polygons = ref_lattice
    return refine_lattice(tiling, n-1)  # recursively call self with one refinement less to do


# computes the variance of the centers of the polygons in the outmost layer
def border_variance(tiling):
    border = []
    mu, var = 0, 0  # mean and variance
    for pgon in [pgon for pgon in tiling.polygons if pgon.sector == 0]:  # find the outmost polygons of sector
        if pgon.layer == tiling.polygons[-1].layer:  # if in highest layer
            mu += weierstrass_distance([0, 0, 1], pgon.centerW)  # [0,0,1] is the origin in weierstrass representation
            border.append(pgon)
    mu /= len(border)  # normalize the mean
    for pgon in border:
        var += (mu-weierstrass_distance([0, 0, 1], pgon.centerW))**2
    return var/len(border)


# the following functions find the total number of polygons for some {p, q} tessellation of l layers
# reference: Baek et al., Phys. Rev.E. 79.011124
def find_num_of_pgons_73(l):
    sum = 0
    s = np.sqrt(5)/2
    for j in range(1, l):
        sum += (3/2+s)**j-(3/2-s)**j
    return int(1+7/np.sqrt(5)*sum)


def find_num_of_pgons_64(l):
    sum = 0
    s = 2*np.sqrt(2)
    for j in range(1, l):
        sum += (3+s)**j-(3-s)**j
    return int(1+s*sum)


def find_num_of_pgons_55(l):
    sum = 0
    s = 3*np.sqrt(5)/2
    for j in range(1, l):
        sum += (7/2+s)**j-(7/2-s)**j
    return int(1+np.sqrt(5)*sum)


def find_num_of_pgons_45(l):
    sum = 0
    s = np.sqrt(3)
    for j in range(1, l):
        sum += (2+s)**j-(2-s)**j
    return int(1+5/s*sum)


def find_num_of_pgons_37(l):
    sum = 0
    s = np.sqrt(5)/2
    for j in range(1, l):
        sum += (3/2+s)**j-(3/2-s)**j
    return int(1+7/np.sqrt(5)*sum)


# the centers (stored in cleanlist) are used to distinguish between polygons
# removing duplicates instantly after their initialization slightly decreases the performance
def remove_duplicates(duplicates, digits=10):
    l = len(duplicates)
    pgonnum = 1
    polygons = []
    centerlist = []
    mid = []
    for pgon in duplicates:
        z = np.round(pgon.centerP, digits)
        if z not in centerlist:
            centerlist.append(z)
            pgon.number = pgonnum
            pgonnum += 1
            polygons.append(pgon)
    print(f"{l - len(centerlist)} duplicate polygons have been removed!")
    return polygons

