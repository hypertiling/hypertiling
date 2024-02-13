Neighbours
==========

This module contains neighbour search algorithms that can be applied to any tiling.

Note that certain construction kernels already built neighbourhood relation upon construction of the tiling or graph! Still, the general functions presented here, can be useful in a number of applications, such as spatial proximity search for refined or manipulated lattices, or in case a range of neighbours larger than immediately close cells is required.

Motivation
----------
Computing adjacency relations in non-Euclidean tilings is challenging, especially for two-dimensional manifolds with negative curvature. These cannot be embedded into higher dimensional Euclidean spaces, making it difficult to establish a natural ordering for grouping or sorting cells by their position. Techniques like partitioning the Poincaré disk into rectangular grids, similar to hierarchical multigrid approaches, are hence ineffective. This is due to the manifold's volume expanding exponentially in all directions, and the coordinate representation becoming significantly distorted near the boundary of the unit circle. Consequently, neither the Poincaré disk coordinates nor any higher-dimensional Euclidean-style grid system is suitable for efficient sorting.


Spatial proximity search
------------------------
Given these difficulties, a conceptually straightforward way of identifying adjacent cells in any regular geometry is by radius search, which is available as a standalone function in the
``neighbors`` module and used as the default algorithm behind `get_nbrs_list` in some kernels, such as DUN07. However, it should be emphasized that any neighbor search method that
makes explicit use of Poincaré disk coordinates risks becoming inaccurate close to the unit circle, where the Euclidean distance between adjacent cells vanishes. For this reason, we recommend performing additional consistency checks whenever very large lattices are required. 

We provide the following functions:

.. automodule:: hypertiling.neighbors
    :members:
