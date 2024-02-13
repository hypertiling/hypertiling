Main Interface
==============

Overview
--------
The core of the ``hypertiling`` package is the ability to generate and manipulate tilings of the hyperbolic plane. These tilings are made up of cells, or polygons in two dimensions, arranged meticulously to cover the entire hyperbolic manifold seamlessly, ensuring no gaps or overlaps. The terminology — tiling, tessellation, and lattice — are used interchangeably throughout this documentation to describe these configurations.

Regular Tilings
---------------
A significant emphasis is placed on regular tilings, which are characterized by their Schläfli symbols, denoted as (p, q). These symbols describe tilings where each polygonal cell has p sides, and each vertex is shared by q cells. For a tiling to be considered hyperbolic, the product (p-2)(q-2) must exceed 4, reflecting the hyperbolic curvature's properties. In these regular tilings, all cells are geometrically congruent, ensuring uniformity across the tiling. For visual examples of these tilings, please refer to Figure 1 in the documentation.

Graph Construction
------------------
Beyond the creation of hyperbolic tilings, the hypertiling package also facilitates the construction of graphs. These graphs serve as simplified, coordinate-free representations of the tilings, focusing solely on the adjacency relationships between the vertices. This functionality broadens the scope of the package, allowing for the exploration of hyperbolic geometry through both visual and structural perspectives.

Interface
---------
Hyperbolic tilings and graphs are generated using two distinct factory patterns. These patterns allow the configuration of basic geometric parameters and the construction kernel. The construction kernel not only determines the algorithm used to construct the object, but also its capabilities and methods:

.. autofunction:: hypertiling.core.HyperbolicTiling

.. autofunction:: hypertiling.core.HyperbolicGraph

