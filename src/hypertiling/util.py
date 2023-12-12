import numpy as np
import math
from .distance import disk_distance


def lattice_spacing_weierstrass(p, q):
    """
    Return the hyperbolic/geodesic lattice spacing h = h^{p,q}, i.e. the edge length of any cell
    """
    num = math.cos(math.pi/q)
    denom = math.sin(math.pi/p)
    return 2*math.acosh(num / denom)


def fund_radius(p, q):
    """
    Radius r_0 of the fundamental polygon in the Poincare disk
    """
    num = math.cos(math.pi*(p+q)/p/q) 
    denom = math.cos(math.pi*(q-p)/p/q)
    return math.sqrt(num / denom)


def cell_radius_weierstrass(p,q):
    """
    Geodesic radius h_r (i.e. distance between center and any vertex) of cells in a regular (p,q) tiling
    This is nothing but the lattice spacing of the dual lattice
    """
    return lattice_spacing_weierstrass(q,p)


def euclidean_center(vertices):
    """
    Compute Euclidean center of a polygon (center of mass)
    """
    vx = np.real(vertices)
    vy = np.imag(vertices)
    return complex(np.mean(vx), np.mean(vy))


def compute_tri_angles(za, zb, zc):
    """
    Use the hyperbolic law of cosines to compute the interiour vertex angles 
    in a triangle given by three points on the Poincare disk, za, zb, zc
    """

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


def n_cell_centered(p,q,n):
    """
    Compute number of polygons in a cell centered regular (p,q) tiling with n layer analytically
    Inspired from Mertens & Moore, PRE 96, 042116 (2017)
    However note that they use a different convention
    """

    retval = 1 # first layer always has one cell
    for j in range(1,n):
        retval = retval + n_cell_centered_recursion(q,p,j) # note the exchange p<-->q
    return retval


def n_cell_centered_recursion(p,q,l):
    """ Helper function """
    a = (p-2)*(q-2)-2
    if l==0:
        return 0
    elif l==1:
        return (p-2)*q
    else:
        return a*n_cell_centered_recursion(p,q,l-1)-n_cell_centered_recursion(p,q,l-2)


def n_vertex_centered(p,q,l):
    """
    Compute number of polygons in a vertex centered regular (p,q) tiling with n layer analytically
    Inspired from Mertens & Moore, PRE 96, 042116 (2017)
    However note that they use a different convention
    """

    if l==0:
        retval = 0 # no faces in zeroth layer
    else:
        retval = ( n_v_vertex_centered(p,q,l)+n_v_vertex_centered(p,q,l-1) )/(p-2)
    return int(retval)


def n_v_vertex_centered(p,q,n):
    """ Helper function """
    retval = 0  # no center vertex without polygons
    for j in range(1,n+1):
        retval = retval + n_cell_centered_recursion(p,q,j)
    return int(retval)
