.. _kernel-dun07:


Generative Reflection (GR) Kernel
=================================

With the GR family, ``hypertiling`` offers a set of
generators that utilize python's generator functions
to save memory and enable larger grids. Here, only one symmetry
sector of the grid is saved and the coordinates of the other
sectors are generated as required.

An important source is
[hyp] "Hypertiling -- a high performance Python library for the generation and visualization of hyperbolic lattices"
(2023, arxiv:2309.10844)

Another difference to the other kernels is the reflective nature
of the kernel. The underlying algorithm for the GR family is different
to the SR (rotational) family. While there new polygons are created
via the rotation of vertices, in the GR family only the corners of
the polygons are reflected. The modified algorithm makes GR
significantly more performant than SR, but forces a modified layer
structure. Rather than layers closing vertices of the previous layer,
the natural definition of layers is given by the edges of the polygons
in the previous layer. Therefore, the next layer closes all open edges
a polygon had.

The actual algorithm is implemented in the GR kernel

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection


Methods
#######

To hide the generative nature, the kernel provides an interface shielding
the user from its generator mechanics. Many of the protected methods, i.e.
methods who's name start with an underscore '_', act on the generative nature.
Methods without underscore hide the generative nature and can be used as with the other kernels.

Integrity
*********

One speciality of the GR-family is the implementation of a '''check_integrity''' method.
The exact tests performed are, even within the GR-family, kernel dependent.
This is necessary as some members are not of subclass tiling and thus do not provide coordinates.
These members are marked with an additional G, e.g. GRG, to indicate its graph nature.

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.check_integrity


Getter methods
**************

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_sector
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_center
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_vertices
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_angle
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_polygon

Layer methods
*************

GR provides methods for accessing the classical layer structure as in the other kernels as well as a method for
accessing its native layer structure. As the classical structure is not natural to GR, it will be calculated on use.
This calculation can either be triggered manually using
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.map_layers
or automatically by the first call of
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_layer

This method will return the native layer definition of GR.
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_reflection_level

Neighbor methods
****************

GR provides multiple methods for accessing the neighbor relations. All of them have their different advantages and
disadvantages. Please see [hyp] for a more detailed performance measurement.

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_nbrs
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_nbrs_generative
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_nbrs_geometrical
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_nbrs_mapping
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.map_nbrs
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_nbrs_radius
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.get_nbrs_list


Miscellaneous methods
*********************

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.generate
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.find


Special methods
***************

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.__len__
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.__iter__
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.__getitem__


Generator methods (protected)
*****************************

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._find
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._get_reflection_level_in_sector
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._get_nbrs_generative
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._get_nbrs_radius
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._get_nbrs_geometrical
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._get_nbrs_mapping


Support methods (protected)
***************************
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._polygen
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._wiggle_index
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._index_from_ref_layer_index
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._expand_sector_index_to_tiling
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection._to_weierstrass


Not implemented methods
***********************
Methods which are not yet implemented but are planned to be added.

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.add_layer
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.transform
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.rotate
.. autoclass:: hypertiling.kernel.GR.GenerativeReflection.translate


Properties
##########
Besides the attributes provides by the tiling-abc, GR provides multiple other attributes. These are all protected and,
due to the generative nature, should be handled with caution.

.. autoclass::hypertiling.kernel.GR.GenerativeReflection._sector_lengths
.. autoclass::hypertiling.kernel.GR.GenerativeReflection._sector_lengths_cumulated
.. autoclass::hypertiling.kernel.GR.GenerativeReflection._sector_polys
.. autoclass::hypertiling.kernel.GR.GenerativeReflection._edge_array
.. autoclass::hypertiling.kernel.GR.GenerativeReflection._sector_lengths
.. autoclass::hypertiling.kernel.GR.GenerativeReflection._layers
