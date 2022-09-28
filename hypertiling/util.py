
import numpy as np
import math
from .distance import weierstrass_distance



# return the hyperbolic/geodesic lattice spacing, i.e. the edge length of any cell
def lattice_spacing_weierstrass(p, q):
    num = math.cos(math.pi/q)
    denom = math.sin(math.pi/p)
    return 2*math.acosh(num / denom)

# radius of the fundamental polygon in the Poincare disk
def fund_radius(p, q):
    num = math.cos(math.pi*(p+q)/p/q)  #np.cos(np.pi / p + np.pi / q)
    denom = math.cos(math.pi*(q-p)/p/q) #np.cos(np.pi / p - np.pi / q)
    return np.sqrt(num / denom)

# geodesic radius (i.e. distance between center and any vertex) of cells in a regular p,q tiling
def cell_radius_weierstrass(p,q):
    # is nothing but the lattice spacing of the dual lattice
    return lattice_spacing_weierstrass(q,p)




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

