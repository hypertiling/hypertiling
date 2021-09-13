import numpy as np
from .distance import weierstrass_distance


def remove_overflow(polygons, num):  # removes excess outmost polygons for given nlayers
    pgons = []
    for ind, pgon in enumerate(polygons):
        pgons.append([abs(pgon.centerP), ind, pgon])
    pgons.sort()
    len_before = len(pgons)
    for rem in range(int(num)): # remove last num elements
        pgons.pop(-1)
    l_after = len(pgons)
    pgons = np.asarray(pgons)
    polygons = []
    for i in range(len(pgons)):
        polygons.append(pgons[i, 2])

    print(f"{len_before-l_after} excess polygons have been removed")
    return polygons


def find_nn_sector(polygons, nn_distance):
    dgts = 10
    adj_mat = np.zeros(shape=(len(polygons), len(polygons)), dtype=np.int32)
    num_of_sectors = polygons[0].p
    k = num_of_sectors / polygons[0].p
    R = round(abs(polygons[1].centerP), dgts)
    neighbors = np.zeros(shape=(len(polygons), int(num_of_sectors)+1+2))  # +3 for info to the corres. polygon
    sectorpgons = [[] for l in range(num_of_sectors)]  # empty list of 2p sublists (one for each sector)
    for row, pgon in enumerate(polygons):  # fill each sector list with polygons of the sector
        pgon.find_angle(k)
        sectorpgons[pgon.sector].append(pgon)
        neighbors[row, 0] = row + 1  # first column is number of the polygon itself
        neighbors[row, 1] = pgon.centerP.real
        neighbors[row, 2] = pgon.centerP.imag

    [row, col] = [0, 1]
    for p1 in polygons:
        sec = p1.sector
        r = round(abs(p1.centerP), dgts)
        if r > R:
            searcharray = sectorpgons[sec] + sectorpgons[sec - 1] + sectorpgons[(sec + 1) % num_of_sectors]
        else:
            searcharray = polygons

        for p2 in searcharray:  # hier evtl. centerW vergleichen
            dist = weierstrass_distance(p2.centerW, p1.centerW)
            if np.round(p1.centerP, dgts) != np.round(p2.centerP, dgts) and round(dist, dgts) == round(nn_distance, dgts):
                neighbors[row, col+2] = p2.number
                adj_mat[p1.number - 1, p2.number - 1] = 1
                col += 1

        row += 1
        col = 1

    # neighbors = np.delete(neighbors, 1, 1)
    # neighbors = np.delete(neighbors, 1, 1)
    # for ind, sublist in enumerate(neighbors[:, 1:]):
    #     sublist.sort()
    #     neighbors[ind, 1:] = sublist

    print(f"Zero as last neighbor in {np.count_nonzero(neighbors[:, -1]==0)} lines")
    return neighbors, adj_mat


# this is implemented poorly and thus rather slow -> bottleneck for large nLayers
# i might be replaced by row or vice versa
def find_nn(polygons, nn_distance):  # compare by element with find_nn_sector! M-M' != 0
    neighbors = np.zeros(shape=(len(polygons), polygons[0].p + 1), dtype=np.int32)  # contains the j-th neighbor of the i-th polygon
    [col, row] = [1, 0]
    for i, p1 in enumerate(polygons):
        for p2 in polygons:
            neighbors[i, 0] = i + 1  # first column is number of the polygon itself
            dist = weierstrass_distance(p2.centerW, p1.centerW)
            if p1.centerP != p2.centerP and round(dist, 9) <= round(nn_distance, 9):  # rounding might be necessary
                neighbors[row, col] = p2.number
                col += 1
        row += 1
        col = 1

    return neighbors


# Finds duplicates right after initialization, currently not in use
# can be merged with remove_duplicates() to one function
# slower than remove_duplicates
def isduplicate(polygon, centerlist):
    # isduplicate.counter += 1
    dgts = 9  # empirically
    pgonnum = 1
    dupnum = 0
    center = round(polygon.centerP, dgts)
    if center not in centerlist:
        duplicate = False
        centerlist.append(center)
        pgonnum += 1
        polygon.number = pgonnum
    else:
        duplicate = True
        dupnum += 1

    # print(f"{dupnum} polygons have been removed by isduplicate().")
    return [duplicate, centerlist]
