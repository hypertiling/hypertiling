import numpy as np


def radial_distance_polar(z):
    return np.abs(z)


# quick and dirty method to locate the "maximal" radial cutoff of a tiling
# returns a boolean array which carries the information whether a node is a boundary node 
# and the actual cutoff radius that was being found and used

# improve me
def maximal_radial_cutoff(T, eps=1e-10):
    boundary = np.zeros(len(T)).astype("bool")

    maxlayer_centers = []
    for j in range(len(T)):
        maxlayer_centers.append(radial_distance_polar(T.get_center(j)))

    cutoff_radius = np.min(maxlayer_centers) - eps

    for j in range(len(T)):
        if radial_distance_polar(T.get_center(j)) > cutoff_radius:
            boundary[j] = True

    return boundary, cutoff_radius



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