
 # Work in progress (TODO)

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