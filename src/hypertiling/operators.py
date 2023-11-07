import numpy as np
from scipy.sparse import coo_matrix

# These are building blocks to construct differential operators on a discrete lattice
# in a finite difference setting

def adjacency(neighbours, weights=None, boundary=None):
    """
    Adjacency matrix of a tiling/graph with optional weights in sparse matrix form

    Arguments:
    ----------
    neighbours : List[List[int]]
        neighbours of a tiling/graph in the hypertiling standard format 
    weights : List[List[float]]
        allows to fill the matrix entries by weights rather than by 0 and 1 alone
    boundary : List[boolean]
        Mask specifying boundary points, which are filtered out (corresponding rows left empty)

    Returns:
    --------
        scipy.sparse.coo_matrix 
    """

    # preparations
    size = len(neighbours)
    rows = np.array([])
    cols = np.array([])
    data = np.array([])
    
    # loop over sites
    for i in range(size):
        # consider only bulk sites
        if boundary is None or not boundary[i]:
            ones = np.ones_like(neighbours[i])
            nbrs = np.array(neighbours[i])      
            
            rows = np.concatenate([rows,i*ones])
            cols = np.concatenate([cols,nbrs])

            if weights is None:
                data = np.concatenate([data,ones])
            else:
                data = np.concatenate([data,weights[i]])      
        
    return coo_matrix((data, (rows, cols)), shape=(size,size))


def degree(neighbours, weights=None, boundary=None):

    """
    Degree matrix of a tiling/graph with optional weights in sparse matrix form

    Arguments:
    ----------
    neighbours : List[List[int]]
        neighbours of a tiling/graph in the hypertiling standard format 
    weights : List[List[float]]
        allows to fill the matrix entries by the sum of connection weights rather than by their number
    boundary : List[boolean]
        Mask specifying boundary points, which are filtered out (corresponding rows left empty)

    Returns:
    --------
        scipy.sparse.coo_matrix 
    """

    # preparations
    size = len(neighbours)
    val = np.zeros(size)
    pos = np.arange(size)

      
    if weights is None:
        for i in range(size):
            val[i] = len(neighbours[i])
    else:
        for i in range(size):
            val[i] = np.sum(weights[i])


    if boundary is None:
        return coo_matrix((val, (pos, pos)), shape=(size,size))
    else:
        bulk = np.invert(boundary)
        bulkval = val[bulk]
        bulkpos = pos[bulk]
        return coo_matrix((bulkval, (bulkpos,bulkpos)), shape=(size,size))


# Identity matrix with optional weights
# points on the boundary are filtered out (rows left empty)

def identity(neighbours, weights=None, boundary=None):

    """
    Identity matrix completing the matrix tool kit

    Arguments:
    ----------
    neighbours : List[List[int]]
        neighbours of a tiling/graph in the hypertiling standard format 
    weights : List[List[float]]
        allows to fill the matrix entries by custom weights rather than by 0 and 1 alone
    boundary : List[boolean]
        Mask specifying boundary points, which are filtered out (corresponding rows left empty)

    Returns:
    --------
        scipy.sparse.coo_matrix 
    """

    # preparations
    size = len(neighbours)
    val = np.zeros(size)
    pos = np.arange(size)

      
    if weights is None:
        val = np.ones(size)
    else:
        for i in range(size):
            val[i] = np.sum(weights[i])

    if boundary is None:
        return coo_matrix((val, (pos, pos)), shape=(size,size))
    else:
        bulk = np.invert(boundary)
        bulkval = val[bulk]
        bulkpos = pos[bulk]
        return coo_matrix((bulkval, (bulkpos,bulkpos)), shape=(size,size))




# The following discretization method is different with respect to how boundary conditions are implemented, 
# but nonetheless equivalent!

def helmholtz_from_hypergraph_sparse(neighbours, mass, q, weight):
    """
    return the discretized Helmholtz operator matrix
    for a graph of constant coordination number q

    use this method if boundary conditions are implemented
    on the right hand side of the resulting linear system
    """

    # preparations
    size = len(neighbours)
    diag = np.arange(size)
    rows = np.array([])
    cols = np.array([])
    data = np.array([])
    
    # non-diagonal entries
    for i in range(size):
        # every entry will be there twice
        # and the coo format automatically sums them up
        # therefore we pass 0.5 times the weight
        a = np.ones(len(neighbours[i]))
        b = np.array(neighbours[i])
        c = 0.5*weight*a
        
        rows = np.concatenate([rows,i*a])
        cols = np.concatenate([cols,b])
        data = np.concatenate([data,c])
        
        rows = np.concatenate([rows,b])
        cols = np.concatenate([cols,i*a])
        data = np.concatenate([data,c])
    
    # diagonal entries (-m-q*W)
    rows = np.concatenate([rows,diag])
    cols = np.concatenate([cols,diag])
    data = np.concatenate([data,(-mass-q*weight)*np.ones(size)])

    return coo_matrix((data, (rows, cols)), shape=(size,size))

