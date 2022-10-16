import copy
import numpy as np
import math

from .distance import weierstrass_distance
from .hyperpolygon import HyperPolygon
from .transformation import p2w
from .geodesics import geodesic_midpoint


# radius of the fundamental (and every other) polygon
def fund_radius(p, q):
    num = math.cos(math.pi*(p+q)/p/q) #np.cos(np.pi / p + np.pi / q)
    denom = math.cos(math.pi*(q-p)/p/q)#np.cos(np.pi / p - np.pi / q)
    return np.sqrt(num / denom)


# returns the polygons of the refined lattice for a given triangular (!) tiling with N polygons
# thus, it returns 4*N polygons
# this can be done faster by once again using symmetry, e.g. with angular_replicate() in core
def refine_lattice(tilingobj, n):  # n is the number of refinements

    if tilingobj.p > 3:
        print("Refinements only work for triangular tilings!")
    if n == 0:  # recursive function terminates for n==0
        return tilingobj  # and returns an instance of the chosen TilingClass

    tiling = copy.deepcopy(tilingobj)  # needed to avoid (I think) pointer issues
    p, q = tiling.p, tiling.q
    ref_lattice = []  # stores the new polygons
    for num, pgon in enumerate(tiling.polygons):  # find the new vertices of each polygon
        ref_vertices = []  # stores newly found vertices through refinement
        # loop through polygon edges
        for vrtx in range(p):
            # find geodesic midpoint
            zm = geodesic_midpoint( pgon.verticesP[vrtx], pgon.verticesP[(vrtx+1)%p] )
            ref_vertices.append(zm)


        # one "mother" triangle bears 4 "children" triangles, one in its mid
        # and three that each share one vertex with their mother

        child = HyperPolygon(p)  # the center triangle whose vertices are the newly found refined ones
        for i in range(p):
            child.verticesP[i] = ref_vertices[i]
        child.verticesP[-1] = pgon.centerP()  # the center triangle shares its center with its mother
        child.idx = 4*num+1  # assigning a unique number
        ref_lattice.append(child)

        for vrtx in range(p):  # for each vertex of the mother triangle that is being refined
            child = HyperPolygon(p)  # these are the non-center children
            vP = [pgon.verticesP[vrtx], ref_vertices[vrtx], ref_vertices[vrtx-1]]
            for i in range(p):
                child.verticesP[i] = vP[i]
            center_x = np.sum(np.real(child.verticesP[:p]))/p  # trick: average over the xs and ys of the vertices to get
            center_y = np.sum(np.imag(child.verticesP[:p]))/p  # ... an approximate value for centerP
            child.verticesP[-1] = complex(center_x, center_y)
            child.idx = (4*num+1)+1+vrtx  # unique number
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




# formula from Mertens & Moore, PRE 96, 042116 (2017)
# note that they use a different convention
def n_cell_centered(p,q,n):
    retval = 1 # first layer always has one cell
    for j in range(1,n):
        retval = retval + n_cell_centered_recursion(q,p,j) # note the exchange p<-->q
    return retval

def n_cell_centered_recursion(p,q,l):
    a = (p-2)*(q-2)-2
    if l==0:
        return 0
    elif l==1:
        return (p-2)*q
    else:
        return a*n_cell_centered_recursion(p,q,l-1)-n_cell_centered_recursion(p,q,l-2)

    
# Eq. A4 from Mertens & Moore, PRE 96, 042116 (2017)
def n_vertex_centered(p,q,l):
  if l==0:
    retval = 0 # no faces in zeroth layer
  else:
    #retval = ( n_v(p,q,l)+n_v(p,q,l-1) )/(p-2)
    retval = ( n_v_vertex_centered(p,q,l)+n_v_vertex_centered(p,q,l-1) )/(p-2)
  return retval

# Eq. A1, A2 from Mertens & Moore, PRE 96, 042116 (2017)
def n_v_vertex_centered(p,q,n):
    retval = 0  # no center vertex without polygons
    for j in range(1,n+1):
        retval = retval + n_cell_centered_recursion(p,q,j)
    return retval



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
        z = np.round(pgon.centerP(), digits)
        if z not in centerlist:
            centerlist.append(z)
            pgon.number = pgonnum
            pgonnum += 1
            polygons.append(pgon)
    print(f"{l - len(centerlist)} duplicate polygons have been removed!")
    return polygons

