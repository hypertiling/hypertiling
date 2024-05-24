.. _kernel-dun07:


Legacy Dunham
=============

``hypertiling`` provides a mostly literal, unoptimized, implementation of the tiling algorithm by Douglas Dunham, translated to Python; More specifically, this is the improved version published in [Dun07]

Sources:

* [Dun86] Dunham, Douglas. "Hyperbolic symmetry." Symmetry. Pergamon, 1986. 139-153.

* [Dun07] Dunham, Douglas. "An algorithm to generate repeating hyperbolic patterns." the Proceedings of ISAMA (2007): 111-118.

* [Dun09] Dunham, Douglas. "Repeating Hyperbolic Pattern Algorithms — Special Cases." unpublished (2009)

* [JvR12] von Raumer, Jakob. "Visualisierung hyperbolischer Kachelungen", Bachelor thesis (2012), unpublished

The actual algorithm is implemented in the DUN07 kernel

.. autoclass:: hypertiling.kernel.DUN07.Dunham

For the convenience of the reader, we present the details of the kernel in separate sections:

.. toctree::
   :maxdepth: 2

   kernel_dun07_details
   kernel_dun07_helpers


.. raw:: html

   <hr />


A central ingredient in this algorithm is the transformation which is successively iterated as the tiling is generated. We wrap it in a class object:

.. autoclass:: hypertiling.kernel.DUN07.DunhamTransformation
    :members:


Getter
------

Finally, the DUN07 kernel satisfies the required getter methods:

.. autofunction:: hypertiling.kernel.DUN07.Dunham.get_polygon
.. autofunction:: hypertiling.kernel.DUN07.Dunham.get_vertices
.. autofunction:: hypertiling.kernel.DUN07.Dunham.get_center
.. autofunction:: hypertiling.kernel.DUN07.Dunham.get_angle


