.. _kernel-dun07:


Generative Reflection (GR) Kernel
=============

With the GR family, ``hypertiling`` offers a set of
generators that utilize python's generator functions
to save memory and enable larger grids. Here, only one symmetry
sector of the grid is saved and the coordinates of the other
sectors are generated as required.

Another difference to the other kernels is the reflective nature
of the kernel. The underlying algorithm for the GR family is different
to the SR (rotational) family. While there new polygons are created
via the rotation of vertices, in the GR family only the corners of
the polygons are reflected. The modified algorithm makes GR
significantly more performant than SR, but forces a modified layer
structure.

The actual algorithm is implemented in the GR kernel

.. autoclass:: hypertiling.kernel.GR.GenerativeReflection
