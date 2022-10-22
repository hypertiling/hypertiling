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
    for poly in T:
        if poly.layer == T.nlayers:
            maxlayer_centers.append(radial_distance_polar(poly.centerP()))

    cutoff_radius = np.min(maxlayer_centers) - eps

    for j in range(len(T)):
        if radial_distance_polar(T[j].centerP()) > cutoff_radius:
            boundary[j] = True

    return boundary, cutoff_radius
