.. Copyright (c) 2022, Manuel Schrauth, Florian Goth

Getting Started with Hypertiling
=================================

In Python, import tiling object from *hypertiling* library

::

   from hypertiling import HyperbolicTiling

Set parameters, initialize and generate the tiling

::

   p = 7
   q = 3
   nlayers = 5

   T = HyperbolicTiling(p,q,nlayers) 
   T.generate()

