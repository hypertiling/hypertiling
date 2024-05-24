Main Interface
==============

Overview
--------
The core of the ``hypertiling`` package is its ability to generate and manipulate tilings of the hyperbolic plane. These tilings consist of cells, or polygons in two dimensions, precisely arranged to cover the entire hyperbolic manifold seamlessly, with no gaps or overlaps. The terms "tiling", "tessellation" and "lattice" are used interchangeably throughout this documentation to describe these configurations.

Regular Tilings
---------------
A significant emphasis is placed on regular tilings, which are characterized by their Schläfli symbols, denoted as (p, q). These symbols describe tilings where each polygonal cell has p sides, and each vertex is shared by q cells. For a tiling to be considered hyperbolic, the product (p-2)(q-2) must exceed 4. In these regular tilings, all cells are geometrically congruent, ensuring uniformity across the tiling.

Graph Construction
------------------
Beyond the creation of hyperbolic tilings, the hypertiling package also facilitates the construction of graphs. These graphs serve as simplified, coordinate-free representations of the tilings, focusing solely on the adjacency relationships between the vertices. 

Interface
---------
Hyperbolic tilings and graphs are generated using two distinct factory patterns. These patterns allow for the configuration of the basic geometric parameters and of the construction kernel. The construction kernel not only determines the algorithm used to construct the object, but also its capabilities and methods:

.. autofunction:: hypertiling.core.HyperbolicTiling

.. autofunction:: hypertiling.core.HyperbolicGraph

