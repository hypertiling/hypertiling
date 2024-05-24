Arithmetics
===========

This documentation page deals with the internal arithmetic functions in the ``hypertiling`` package. Since those functions are at the core of the package, some of them are heavily optimized, using e.g. numba just-in-time compilation and furthermore particularly designed to be numerically stable, given the coordinate singularities and largely varying coordinate dimensions that come with the Poincare disk representation of hyperbolic geometry. Some of the transformations are available in double-double precision. 


Kahan summation
---------------

Kahan summation algorithm, also known as compensated summation, is a method to improve the numerical accuracy of the total sum of a sequence of finite precision floating point numbers. The algorithm significantly reduces the numerical error in the total obtained by adding a sequence of numbers compared to the simple approach of adding them in sequence. This is particularly important in cases where the sum involves a large number of terms or when the numbers vary greatly in magnitude.

.. autofunction:: hypertiling.arithmetics.kahan

.. raw:: html

   <hr />
   
Branch-free transformations
---------------------------
Branch-free summation and differences, as described, e.g., by Donald Knuth, is an approach used in computer programming and numerical analysis to perform addition operations without causing conditional branches. Conditional branches can be expensive in terms of CPU cycles, especially on modern processors with deep pipelines and speculative execution. Branch mispredictions can lead to significant performance penalties, so eliminating branches, when possible, can lead to more efficient code execution.
    
.. autofunction:: hypertiling.arithmetics.twosum
.. autofunction:: hypertiling.arithmetics.twodiff

.. raw:: html

   <hr />

Double-double precision
-----------------------

Double-double precision is a technique used in computing to achieve higher precision than what is provided by a standard double-precision floating-point format. While a standard double-precision floating-point number uses 64 bits to represent a number, double-double precision effectively uses two 64-bit double-precision numbers to represent a single number with greater precision. This technique allows for a significant increase in the number of significant digits that can be accurately represented.

.. autofunction:: hypertiling.arithmetics.htadd
.. autofunction:: hypertiling.arithmetics.htdiff
.. autofunction:: hypertiling.arithmetics.htprod
.. autofunction:: hypertiling.arithmetics.htcplxprod
.. autofunction:: hypertiling.arithmetics.htcplxprodconjb
.. autofunction:: hypertiling.arithmetics.htcplxadd
.. autofunction:: hypertiling.arithmetics.htcplxdiff
.. autofunction:: hypertiling.arithmetics.htcplxdiv