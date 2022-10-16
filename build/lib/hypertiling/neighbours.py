import numpy as np
import math
from .distance import weierstrass_distance, lorentzian_distance

# wrapper to provide a nicer interface
def find(tiling, nn_dist=None, which="optimized", index_from_zero=True, verbose=False):

    if nn_dist == None:
        if verbose:
            print("No search radius given; using distance between first and second vertex in the tessellation!")
        nn_dist = weierstrass_distance(tiling[0].centerW(), tiling[1].centerW())

    if which == "optimized_slice":
        retval = find_nn_optimized_slice(tiling, nn_dist) # fastest
    elif which == "optimized":
        retval = find_nn_optimized(tiling, nn_dist)
    elif which == "brute_force":
        retval = find_nn_brute_force(tiling, nn_dist) # use for debug
    elif which == "slice":
        retval = find_nn_slice(tiling, nn_dist)
    elif which == "edge_map":
        retval = find_nn_edge_map_optimized(tiling)
    elif which == "edge_map_brute_force":
        retval = find_nn_edge_map_brute_force(tiling)

    else:
        print("[Hypertiling] Error:", which, " is not a valid algorithm!")
    

    nbrs = []
    if index_from_zero:
        for sublist in retval:
            new_sublist = [x-1 for x in sublist]
            nbrs.append(new_sublist)
                
        return nbrs
    else:
        return retval



# find neighbours by identifying corresponding edges among polygons
# this is a coordinate-free algorithm, it uses only the graph structure
# can probably be further improved
def find_nn_edge_map_optimized(tiling):

    tiling.populate_edge_list()

    # we create a kind of "dictionary" where keys are the
    # edges and values are the corresponding polygon indices
    edges, vals = [], []
    for poly in tiling:
        for edge in poly.edges:
            edges.append(edge)
            vals.append(poly.idx)

    # reshape "edges" into its components in order to 
    # make use of numpy vectorization later
    edge0 = np.zeros(len(edges)).astype(complex)
    edge1 = np.zeros(len(edges)).astype(complex)
    for i, edge in enumerate(edges):
        edge0[i] = edge[0]
        edge1[i] = edge[1]
    
    # create empty neighbour array
    nbrs = []
    for i in range(len(tiling)):
        nbrs.append([])

    # an edge that is share by two polygons appears twice
    # in the "edges" list; we find the corresponding polygon
    # indices by looping over that list
    for i, edge in enumerate(edges):
        # compare against full edges arrays
        # this avoids a double loop which is slow ...
        # check edge
        bool_array1 = (edge[0] == edge0)
        bool_array2 = (edge[1] == edge1)
        # check also reverse orientation
        bool_array3 = (edge[0] == edge1)
        bool_array4 = (edge[1] == edge0)
        
        # put everything together; we require 
        # (True and True) or (True and True)
        b = bool_array1*bool_array2 + bool_array3*bool_array4
        
        # find indices where resulting boolean array is true
        w = np.where(b)
        
        # these indices are neighbours of each other
        for x in w[0]:
            if vals[i] is not vals[x]:
                nbrs[vals[i]-1].append(vals[x])    
    
    return nbrs


# find neighbours by identifying corresponding edges among polygons
# there is an equivalent method available that is much faster: find_nn_edge_map_optimized
# use this method only for debugging purposes
def find_nn_edge_map_brute_force(tiling):

    tiling.populate_edge_list()

    # we create a kind of "dictionary" where keys are the
    # edges and values are the corresponding polygon indices
    edges, vals = [], []
    for poly in tiling:
        for edge in poly.edges:
            edges.append(edge)
            vals.append(poly.idx)

    # create empty neighbour array       
    nbrs = []
    for i in range(len(tiling)):
        nbrs.append([])

    # an edge that is share by two polygons appears twice
    # in the "edges" list; we find the corresponding polygon
    # indices by looping over that list twice
    for i, k1 in enumerate(edges):
        for j, k2 in enumerate(edges):
            if k1[0] == k2[0] and k1[1] == k2[1]:  
            # check edge      
                if vals[i] is not vals[j]:
                    nbrs[vals[i]-1].append(vals[j])
            # check also reverse orientation    
            elif k1[1] == k2[0] and k1[0] == k2[1]: 
                if vals[i] is not vals[j]:
                    nbrs[vals[i]-1].append(vals[j])

    
    return nbrs



# find nearest neighbours by brute force comparison of all-to-all distances
# scales quadratically in the number of vertices and may thus become prohibitively expensive
# might be used for debugging purposes though
def find_nn_brute_force(tiling, nn_dist, eps=1e-8):
    retlist = []  # prepare list
    for poly1 in tiling.polygons:  # loop over polygons
        sublist = []
        for poly2 in tiling.polygons:
            dist = weierstrass_distance(poly1.centerW(), poly2.centerW()) # compare distances
            if dist < nn_dist + eps: # add something to nn_dist to avoid rounding problems
                if poly1.idx is not poly2.idx   :  # avoiding finding A as neighbor of A
                    sublist.append(poly2.idx)
        retlist.append(sublist)
    return retlist



# finds nearest neighbours by comparing all-to-all distances
# however, making sure everything can be fully vectorized by numpy we gain a significant speed-up
def find_nn_optimized(tiling, nn_dist, eps=1e-5):
    # prepare matrix containing all center coordiantes
    v = np.zeros((len(tiling), 3))
    for i, poly in enumerate(tiling):
        v[i] = poly.centerW()

    # add something to nn_dist to avoid rounding problems
    # does not need to be particularly small
    searchdist = nn_dist + eps
    searchdist = math.cosh(searchdist)

    # prepare list
    retlist = []

    # loop over polygons
    for i, poly in enumerate(tiling):
        w = poly.centerW()
        dists = lorentzian_distance(v, w)
        dists[(dists < 1)] = 1  # this costs some %, but reduces warnings
        indxs = np.where(dists < searchdist)[0]  # radius search
        self = np.argwhere(indxs == i)  # find self
        indxs = np.delete(indxs, self)  # delete self
        nums = [tiling[ind].idx for ind in indxs]  # replacing indices by actual polygon number
        retlist.append(nums)
    return retlist


def find_nn_optimized_slice(tiling, nn_dist, eps=1e-5):
    """
    combines both the benefits of of numpy vectorization (used in "find_nn_optimized") and 
    applying the neighbour search only to a p-sector of the tiling (as done in "find_nn_slice")
    currently this is our fastest algorithm for large tessellations

    currently only working for cell-centered tilings (to do!)

    Attributes
    ----------

    tiling : HyperbolicTiling
        the tiling object in which adjacent cell are to be searched for
    nn_dist : float
        the expected distance between neighbours
    eps : float
        increase nn_dist a little in order to make it more stable

    Returns
    -------

    List of lists, where sublist i contains the indices of the neighbour of vertex i in the tiling 

    """


    if tiling.center == "vertex":
        print("algorithm >optimized_slice< currently not available for vertex-centered tilings!")
        print("falling back to >optimized<")
        return find_nn_optimized(tiling, nn_dist)

    totalnum = len(tiling)		# total number of polygons
    p = tiling.p             	# number of edges of each polygon
    q = tiling.q
    if tiling.center == "cell":
        pps = int((totalnum-1)/p)	# polygons per sector (excluding center)
        inc = 1
    elif tiling.center == "vertex":
        pps = int(totalnum/q)
        inc = 0
    
    
    # shifts local neighbour list to another sector
    def shift(lst,times):
    	lst = sorted(lst)
    	haszero = (0 in lst)
       
        # fundamental cell requires a little extra care
    	if haszero:
    		lsta = np.array(lst[1:]) + times*pps
    		lsta[lsta>totalnum-1] -= (totalnum-1)
    		lsta[lsta<1] += (totalnum-1)
    		return sorted([0]+list(lsta))
    	else:
    		lsta = np.array(lst) + times*pps
    		lsta[lsta>totalnum-1] -= (totalnum-1)
    		lsta[lsta<1] += (totalnum-1)
    		return sorted(list(lsta))
    
    # increment indices by one
    def increment(lst):
    	return list(np.array(lst)+1)

    
    
    # slice the first three sectors
    # we are gonna look for neighbours of polygons in the second sector 
    pgons = tiling[:3*pps+inc]
    # this is a place where the algorithm can be further improved, performance-wise
    # we do not need the entire 1st and 3rd sectors, but only those cells close to 
    # the boundary of the 2nd sector
    
    
    # store center coordinates in array for faster access
    v = np.zeros((len(pgons), 3))
    for i, poly in enumerate(pgons):
    	v[i] = poly.centerW()
    
    # the search distance (we are doing a radius search)
    searchdist = nn_dist + eps
    searchdist = np.cosh(searchdist)
        
    # prepare list
    nbrlst = []
    # loop over polygons
    for i, poly in enumerate(pgons[pps+inc:2*pps+inc]):
    	w = poly.centerW()
    	dists = lorentzian_distance(v, w)
    	dists[(dists < 1)] = 1  # this costs some %, but reduces warnings
    	indxs = np.where(dists < searchdist)[0]  # radius search
    	self = np.argwhere(indxs == poly.idx-1)  # find self
    	indxs = np.delete(indxs, self)  # delete self
    	nums = [pgons[ind].idx-1 for ind in indxs]  # replacing indices by actual polygon number
    	nbrlst.append(nums)
    
    
    # prepare full output list
    retlist = []
    
    
    if tiling.center == "cell":
        k = p
    elif tiling.center == "vertex":
        k = q
    
    
    if tiling.center == "cell":
        # fundamental cell
        lstzero = []
        for ps in range(0,k):
        	lstzero.append(ps*pps+1)
        retlist.append((increment(lstzero)))
    
    # first sector
    for lst in nbrlst:
    	retlist.append(increment(shift(lst, -1)))
    
    # second sector
    for lst in nbrlst:
    	retlist.append(increment(lst))
    
    # remaining sectors
    for ps in range(2,k):
    	for lst in nbrlst:
    		retlist.append(increment(shift(lst, ps-1)))
    
    return retlist




# finds nearest neighbors by exploiting the discrete rotational symmetry of the lattice
# the neighbors of a polygon in some sector s can be found by adding the number of polygons 
# in that sector to the corresponding number of each neighbor of the polygon at the same position 
# in sector-s. Thus, only in sector one has to find neighbors, the neighbors of every other polygon can be calculated
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
                dist = weierstrass_distance(pgon.centerW(), polygon.centerW())
                if polygon.centerP != pgon.centerP and round(dist, 9) <= round(nn_distance, 9):
                    nn_sector[row, col] = pgon.idx
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
