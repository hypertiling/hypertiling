
# Hypertiling

This is a Python 3 libary for the fast generation of regular hyperbolic tilings, embedded in the Poincare disk model.

Using efficient algorithms and the power of numpy, hyperbolic graphs with millions of polygons can be created in a matter of minutes on a single CPU. We also provide optimized search algorithms for finding adjacent vertices, which allows to use the graph for all sorts of scientific purposes.

<p align="center">
  <img src="assets/hyp6.svg" width="500" />
</p>

# Installation

Hypertiling is not yet available for `pip`, but can be locally built using
```
$ python setup.py install
```

For developer mode use
```
$ python setup.py develop
```
