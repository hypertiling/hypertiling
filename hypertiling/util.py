import numpy as np
import math
from .distance import weierstrass_distance, disk_distance


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


# compute Euclidean center of a polygon (center of mass)
def euclidean_center(vertices):
    vx = np.real(vertices)
    vy = np.imag(vertices)
    return complex(np.mean(vx), np.mean(vy))


# use the hyperbolic law of cosines to compute the interiour vertex angles in a triangle
# given by three points za, zb, zc
def compute_tri_angles(za, zb, zc):

    # compute edge lengths
    a = disk_distance(zb,zc)
    b = disk_distance(za,zc)    
    c = disk_distance(za,zb)

    # pre-compute cosh/sinh
    cosha = np.cosh(a)
    coshb = np.cosh(b)
    coshc = np.cosh(c)
    sinha = np.sinh(a)
    sinhb = np.sinh(b)
    sinhc = np.sinh(c)

    # apply law of cosines
    cosgamma = (coshc - cosha*coshb) / (sinha*sinhb)
    cosalpha = (cosha - coshc*coshb) / (sinhc*sinhb)
    cosbeta  = (coshb - cosha*coshc) / (sinha*sinhc)

    # alpha is the angle opposite of edge "a", etc.
    return np.arccos(cosalpha), np.arccos(cosbeta), np.arccos(cosgamma)





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
  return int(retval)

# Eq. A1, A2 from Mertens & Moore, PRE 96, 042116 (2017)
def n_v_vertex_centered(p,q,n):
    retval = 0  # no center vertex without polygons
    for j in range(1,n+1):
        retval = retval + n_cell_centered_recursion(p,q,j)
    return int(retval)


