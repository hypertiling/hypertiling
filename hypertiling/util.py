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




 

def numerically_unstable_upper(self, l, start, end, tolfactor=10, samplesize=10):
    """
    check whether the true "embedding" distance between cells in layer l comes close
    to the rounding accuracy
    """

    # innermost layers are always fine, do nothing
    if l<3:
        return False

    # randomly pick a number of sites from l-th layer
    curr_layer = self.polygons[start:end]
    layersize = end-start
    true_dists = []

    for i in range(samplesize):
        rndidx = np.random.randint(layersize)

        # generate an adjacent cell
        mother = curr_layer[rndidx]
        child  = self.generate_adj_poly(copy.deepcopy(mother), 0, 1)

        # compute the true (non-geodesic) distance
        true_dist = np.abs(mother.centerP()-child.centerP())
        true_dists.append(true_dist)

    # if this distances comes close to the rounding accuracy
    # two cells can no longer be reliably distinguished
    if np.min(true_dist) < self.accuracy*tolfactor:
        return True
    else:
        return False


def numerically_unstable_lower(self, l, start, end, tolfactor=10, samplesize=100):
    """
    we know which geodesic distance two adjancent cells are supposed to have;
    here we take a sample of cells from the l-th layer and compute mutual 
    distances; if one of those is significantly off compared to the expected
    value we are about to enter a dangerous regime in terms of rounding errors
    """

    # innermost layers are always fine, do nothing
    if l<3:
        return False

    # take a sample of cells and compute their distances
    samples = self.polygons[start:end][:samplesize]
    disk_distances = []
    for j1, pgon1 in enumerate(samples):
        for j2, pgon2 in enumerate(samples):
            if j1 != j2:
                disk_distances.append(disk_distance(pgon1.centerP(), pgon2.centerP()))

    # we are interested in the minimal distance (can be interpreted as an 
    # upper bound on the accumulated error)
    mindist = np.min(np.array(disk_distances))

    # the reference distance
    refdist = disk_distance(self.fund_poly.centerP(), self.polygons[1].centerP())

    # if out arithmetics worked error-free, mindist = refdist
    # in practice, it does not, so we compute the difference
    # if it comes close to the rounding accuracy, adjacency can no longer
    # by reliably resolved and we are about to enter a possibly unstable regime
    if np.abs(mindist-refdist) > self.accuracy/tolfactor:
        return True
    else:
        return False

