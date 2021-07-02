import numpy as np
from .distance import weierstrass_distance
import time


# find nearest neighbours by brute force comparison of all-to-all distances.
# Scales quadratically in the number of vertices and may thus become prohibitively expensive
def find_nn_brute_force(tiling, nn_dist):
    increment = 0.001  # add something to nn_dist to avoid rounding problems. Has to be smaller than the fund_rad
    retlist = []  # prepare list
    for i, poly1 in enumerate(tiling.polygons):  # loop over polygons
        sublist = []  # dummy
        for j, poly2 in enumerate(tiling.polygons):
            dist = weierstrass_distance(poly1.centerW, poly2.centerW) # compare distances
            if dist < nn_dist+increment:
                if i is not j:  # avoiding finding A as neighbor of A
                    sublist.append(j)
        retlist.append(sublist)
    return retlist


# finds nearest neighbours by comparing all-to-all distances
# however, making sure everything can be fully vectorized by numpy we gain a significant speed-up
def find_nn_optimized(tiling, nn_dist):
    # prepare
    v0 = np.zeros(len(tiling.polygons))
    v1 = np.zeros(len(tiling.polygons))
    v2 = np.zeros(len(tiling.polygons))
    for i, poly in enumerate(tiling):
        v0[i] = poly.centerW[0]  # x
        v1[i] = poly.centerW[1]  # y and
        v2[i] = poly.centerW[2]  # z coordinate in weierstrass representation

    increment = 0.001  # add something to nn_dist to avoid rounding problems
    searchdist = nn_dist + increment
    retlist = []  # prepare list
    for i, poly1 in enumerate(tiling):  # loop over polygons
        vA = poly1.centerW  # distance step 1
        args = vA[2] * v2 - vA[1] * v1 - vA[0] * v0
        args[(args < 1)] = 1  # this costs some %, but reduces warnings
        dists = np.arccosh(args)  # distance step 2; this uses the vectorization of numpy functions
        indxs = np.where(dists < searchdist)[0]  # radius search
        self = np.argwhere(indxs == i)  # find self
        indxs = np.delete(indxs, self)  # delete self
        nums = [tiling[ind].number for ind in indxs]  # replacing indices by actual polygon number
        retlist.append(nums)
    return retlist


# finds nearest neighbors by capitalizing on the fact that the neighbors of a polygon in some sector s can be found
# by adding the number of polygons in that sector to the corresponding number of each neighbor of the polygon at the
# same position in sector-s. Thus, only in sector one has to find neighbors (done with find_nn_optimized), the neighbors
# of every other polygon can be calculated.
# exploits rotational symmetry
# since find_nn_optimized scales exponentially(?), this boosts performance drastically. For {7,3}-10 with 29261
# polygons, this function was 14x faster than find_nn_optimized (29.7s vs 2.1s)
def find_nn_optimized_slice(tiling, nn_dist):
    pgons = []
    for pgon in tiling.polygons:  # pick those polygons that are in sector 0, 1 or last (3 adjacent sectors)
        pgon.find_angle(1)
        if pgon.sector in [0, 1, tiling.polygons[0].p-1]:
            pgons.append(pgon)
    # prepare
    v0 = np.zeros(len(pgons))
    v1 = np.zeros(len(pgons))
    v2 = np.zeros(len(pgons))
    for i, poly in enumerate(pgons):
        v0[i] = poly.centerW[0]  # x
        v1[i] = poly.centerW[1]  # y and
        v2[i] = poly.centerW[2]  # z coordinate in weierstrass representation

    increment = 0.001  # add something to nn_dist to avoid rounding problems. Has to be less than fund_radius
    searchdist = nn_dist + increment
    retlist = []  # prepare list
    for i, poly1 in enumerate(pgons):  # loop over polygons
        if poly1.sector == 0:
            vA = poly1.centerW  # distance step 1
            args = vA[2] * v2 - vA[1] * v1 - vA[0] * v0
            args[(args < 1)] = 1  # this costs some %, but reduces warnings
            dists = np.arccosh(args)  # distance step 2; this uses the vectorization of numpy functions
            indxs = np.where(dists < searchdist)[0]  # radius search
            self = np.argwhere(indxs == i)  # find self
            indxs = np.delete(indxs, self)  # delete self
            nums = [pgons[ind].number for ind in indxs]  # replacing indices by actual polygon number
            retlist.append(nums)

    pps = int((len(pgons)-1)/3)  # polygons per sector, excl. center polygon
    p = pgons[0].p  # number of edges of each polygon
    total_num = p*pps+1  # total number of polygons

    lst = np.zeros((pps+1, p+1), dtype=np.int32)  # fundamental nn-data of sector 0
    lst[:, 0] = np.linspace(1, pps+1, pps+1)  # writing no. of each polygon in the first column
    for ind, line in enumerate(retlist):  # padding retlist with zeros such that each array has the same length
        lst[ind, 1:len(line)+1] = line

    neighbors = np.zeros(shape=(total_num, p + 1), dtype=np.int32)  # this array stores nn-data of the whole tiling
    neighbors[:pps+1] = lst
    nn_of_first = [2]  # the first polygon has to be treated seperately
    for n in range(1, p):
        # the crucial steps of this function follow: increase both the number of the polygon and of its neighbors by the
        # number of polygons in this sector to obtain the result for the adjacent sector. Done, p-1 times, this returns
        # the neighbors of the whole tiling
        lst[(lst > 0)] += pps  # increase each entry by pps if its non-zero
        nn_of_first.append(nn_of_first[-1] + pps)  # increase last nn of first polygon by pps
        for ind, row in enumerate(lst[1:]):  # iterate over every
            row[1] = 1 if ind == 0 else row[1]  # taking care of second layer, special treatment for center polygon
            row[:] = [elem % total_num+1 if elem > total_num else elem for elem in row]
            neighbors[1+ind+n*pps] = row  # fill the corresponding row in the neighbors matrix

    neighbors[0, 1:] = nn_of_first  # finally store the neighbors of center polygon
    return neighbors


def find_nn_slice(slices, nn_distance):  # slices contains polygons of two adjacent slices
    pps = int((len(slices) - 1) / 3) + 1  # polygons per sector, incl. center polygon
    p = slices[0].p
    total_num = p * (pps - 1) + 1
    nn_sector = np.zeros(shape=(pps, p + 1), dtype=np.int32)
    col = 1
    for row, polygon in enumerate(slices):
        if polygon.sector == 0:  # find nn only for polygons of the first 1/p-slice
            nn_sector[row, 0] = row + 1
            for pgon in slices:
                dist = weierstrass_distance(pgon.centerW, polygon.centerW)
                if polygon.centerP != pgon.centerP and round(dist, 9) <= round(nn_distance, 9):
                    nn_sector[row, col] = pgon.number
                    col += 1
            col = 1

    ones = np.ones_like(nn_sector)
    neighbors = np.zeros(shape=(total_num, p + 1), dtype=np.int32)
    neighbors[:pps, :p + 1] = nn_sector
    nn_of_first = [2]
    for n in range(1, p):
        mat = nn_sector + n * (pps - 1) * ones
        nn_of_first.append(nn_of_first[-1] + (pps - 1))
        for ind, row in enumerate(mat[1:, :]):
            row[1] = 1 if ind==0 else row[1]
            row[:] = [elem%total_num+1 if elem>total_num else elem for elem in row]
            row[:] = [0 if elem == n*(pps - 1) else elem for elem in row]
            neighbors[1+ind+n*(pps - 1), :] = row
    neighbors[0, 1:] = nn_of_first
    return neighbors
