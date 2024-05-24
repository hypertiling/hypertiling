Transformations
===============

As a geometry library, ``hypertiling`` relies heavily on transformation functions, such as translations, rotations, reflections, etc, both applied to single vertices, arrays of vertices or cell objects. Since those functions are at the core of the package, some of them are heavily optimized, using e.g. numba just-in-time compilation and furthermore particularly designed to be numerically stable, given the coordinate singularities that come with the Poincare disk representation of hyperbolic geometry. Some of the transformations are internally available in double-double precision. 

Double-double precision is a technique used in computing to achieve higher precision than what is provided by a standard double-precision floating-point format. While a standard double-precision floating-point number uses 64 bits to represent a number, double-double precision effectively uses two 64-bit double-precision numbers to represent a single number with greater precision. This technique allows for a significant increase in the number of significant digits that can be accurately represented.


Moebius Transformations
-----------------------

These are the standard Moebius transformations available in the package:

.. autofunction:: hypertiling.transformation.moeb_origin_trafo
.. autofunction:: hypertiling.transformation.moeb_origin_trafo_inversedd
.. autofunction:: hypertiling.transformation.moeb_origin_trafodd
.. autofunction:: hypertiling.transformation.moeb_rotate_trafo
.. autofunction:: hypertiling.transformation.moeb_rotate_trafodd


.. raw:: html

   <hr />

Array Transformations
---------------------

In the ``arraytransformation`` module we find the corresponding function, acting on arrays of points:

.. automodule:: hypertiling.arraytransformation
    :members: