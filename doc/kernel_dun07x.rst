Improved Dunham
===============

We provide a performance optimized implementation of the tiling algorithm by Douglas Dunham. It  achieves a factor 10-20x speed up compared to the literal, unoptimized implementation, which is available as the :ref:`legacy Dunham kernel <kernel-dun07>`.

Sources:

* [Dun86] Dunham, Douglas. "Hyperbolic symmetry." Symmetry. Pergamon, 1986. 139-153.

* [Dun07] Dunham, Douglas. "An algorithm to generate repeating hyperbolic patterns." the Proceedings of ISAMA (2007): 111-118.

* [Dun09] Dunham, Douglas. "Repeating Hyperbolic Pattern Algorithms — Special Cases." unpublished (2009)

* [JvR12] von Raumer, Jakob. "Visualisierung hyperbolischer Kachelungen", Bachelor thesis (2012), unpublished



The improved Dunham kernel is implemented as:

.. autoclass:: hypertiling.kernel.DUN07X.DunhamX

Kernel Details
--------------

The following function defines a method to calculate the fundamental polygon. It creates an array of complex numbers representing vertices in Weierstrass coordinates, and then converts them to a 3D array. The method returns the 3D array representing the fundamental polygon.

.. autofunction:: hypertiling.kernel.DUN07X.DunhamX._create_fundamental_polygon

This is the driver routine for the Dunham tiling algorithm. It iterates over each vertex of the polygon and, for each vertex, iterates over a certain number of polygons. For each polygon, it determines the exposure level, replicates the motif, and modifies the transformation. It is not part of the class but a separate helper function. Since this kernel is improved with performance in mind, a larger number of function arguments could not be avoided.

.. autofunction:: hypertiling.kernel.DUN07X.generate_dun

What is left is the function which initiates the construction:

.. autofunction:: hypertiling.kernel.DUN07X.DunhamX._generate


.. raw:: html

   <hr />

Helper functions
----------------

As for every kernel, there is a number of helpers:

.. autofunction:: hypertiling.kernel.DUN07X.DunhamX._compute_edge_reflections
.. autofunction:: hypertiling.kernel.DUN07X.comp_tran
.. autofunction:: hypertiling.kernel.DUN07X.add_trans

.. raw:: html

   <hr />


Getter
------

Finally, the DUN07X kernel satisfies the required getter methods:

.. autofunction:: hypertiling.kernel.DUN07X.DunhamX.get_polygon
.. autofunction:: hypertiling.kernel.DUN07X.DunhamX.get_vertices
.. autofunction:: hypertiling.kernel.DUN07X.DunhamX.get_center
.. autofunction:: hypertiling.kernel.DUN07X.DunhamX.get_angle