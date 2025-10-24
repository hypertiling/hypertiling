import numpy as np
import math
from .distance import poincare_distance


def _check_hyperbolic(p: int, q: int) -> None:
    """
    Hyperbolic tilings require (p-2)(q-2) > 4 and p,q >= 3.
    Raises on invalid parameters to avoid NaN domains
    """
    if not (isinstance(p, int) and isinstance(q, int) and p >= 3 and q >= 3):
        raise ValueError("p and q must be integers >= 3.")
    if (p - 2) * (q - 2) <= 4:
        raise ValueError("Regular {p,q} is not hyperbolic when (p-2)(q-2) <= 4.")



# -------- edge lengths for regular polygonal tilings ---------


def edge_length_geodesic(p: int, q: int) -> float:
    """
    Compute the geodesic edge length :math:`h^{(p,q)}` of the regular {p,q} tiling
    in the Poincaré disk model.

    The formula follows from hyperbolic right-triangle relations:
        cosh(h/2) = cos(π/q) / sin(π/p)

    Parameters
    ----------
    p : int
        Number of edges (sides) of each polygonal cell.
    q : int
        Number of polygons meeting at each vertex.

    Returns
    -------
    float
        Geodesic edge length :math:`h^{(p,q)}` between adjacent vertices.
    """
    _check_hyperbolic(p, q)
    num = math.cos(math.pi / q)
    denom = math.sin(math.pi / p)
    val = num / denom
    if val < 1:
        # Shouldn't happen for hyperbolic {p,q}, but guard for numerics.
        val = 1.0
    return 2.0 * math.acosh(val)



def dual_edge_length_geodesic(p: int, q: int) -> float:
    """
    Compute the geodesic edge length :math:`h^{(q,p)}` of the dual tiling {q,p}
    in the Poincaré disk model.

    Two equivalent formulations exist:

    1. By definition, the dual tiling is {q,p}, hence
           h_dual = h({q,p})
    2. Geometrically, adjacent cell centers lie on a geodesic orthogonal
       to the shared edge and are separated by twice the inradius, i.e.
           h_dual = 2 * r

    Both definitions yield the same result.

    Parameters
    ----------
    p : int
        Number of edges (sides) of each polygonal cell in the primal tiling.
    q : int
        Number of polygons meeting at each vertex.

    Returns
    -------
    float
        Geodesic edge length :math:`h^{(q,p)}` of the dual tiling.
    """
    return edge_length_geodesic(q, p)



# -------- cell radius for regular polygonal tilings ---------

def cell_radius_geodesic(p: int, q: int) -> float:
    """
    Compute the geodesic circumradius :math:`h_r` of a {p,q} cell in the Poincaré disk.

    The circumradius is the hyperbolic distance from the polygon center
    to any of its vertices and satisfies:
        cosh R = cot(π/p) * cot(π/q)

    Parameters
    ----------
    p : int
        Number of edges (sides) of each polygonal cell.
    q : int
        Number of polygons meeting at each vertex.

    Returns
    -------
    float
        Geodesic circumradius :math:`h_r` (center → vertex).
    """

    _check_hyperbolic(p, q)
    val = (math.cos(math.pi / p) / math.sin(math.pi / p)) * (math.cos(math.pi / q) / math.sin(math.pi / q))
    if val < 1:
        val = 1.0
    return math.acosh(val)


def fundamental_radius(p: int, q: int) -> float:
    """
    Compute the Euclidean radius :math:`r_0` of a {p,q} cell’s circumcircle
    in the Poincaré disk model (center → vertex).

    Conversion from hyperbolic circumradius R to Euclidean radius r0:
        r0 = tanh(R / 2)

    Parameters
    ----------
    p : int
        Number of edges (sides) of each polygonal cell.
    q : int
        Number of polygons meeting at each vertex.

    Returns
    -------
    float
        Euclidean radius :math:`r_0` in disk coordinates (0 < r0 < 1).
    """ 

    R = cell_radius_geodesic(p, q)
    return math.tanh(R / 2.0)



def inradius_geodesic(p: int, q: int) -> float:
    """
    Compute the geodesic inradius :math:`r` of a {p,q} cell in the Poincaré disk.

    The inradius is the hyperbolic distance from the cell center to the midpoint
    of any edge and satisfies:
        cosh r = cos(π/p) / sin(π/q)

    Parameters
    ----------
    p : int
        Number of edges (sides) of each polygonal cell.
    q : int
        Number of polygons meeting at each vertex.

    Returns
    -------
    float
        Geodesic inradius :math:`r` (center → edge midpoint).
    """
    _check_hyperbolic(p, q)
    val = math.cos(math.pi / p) / math.sin(math.pi / q)
    if val < 1:
        val = 1.0
    return math.acosh(val)



# -------- geometric helpers ---------

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
    a = poincare_distance(zb,zc)
    b = poincare_distance(za,zc)    
    c = poincare_distance(za,zb)

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



# -------- projected number of polygons ---------


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
