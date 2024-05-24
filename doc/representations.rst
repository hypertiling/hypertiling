Representations
===============

Since we use both the Poincare disk as well as the Weierstrass (or hyperboloid) representation of the hyperbolic plane, transformations between the two coordinate systems are required. 

Note that the Weierstrass coordinate system can be represented in two different orders: (t, x, y) and (x, y, t). These should not be confused.

The Hyperbolic Weierstrass coordinates, also known as hyperboloid coordinates, are used in hyperbolic geometry to describe points on a hyperboloid in Minkowski space. In this system, a point on the hyperboloid is given by coordinates that satisfy the relation:

.. math::

   -t^2 + x^2 + y^2 = -1

for the one-sheeted hyperboloid in three dimensions. This coordinate system describes points on a hyperboloid in Minkowski spacetime. The Weierstrass coordinates provide a natural way to parametrize hyperbolic space, facilitating calculations in areas such as Lorentz transformations and hyperbolic trigonometry. These coordinates are particularly useful for visualizing the geometry of the hyperbolic plane and for performing complex transformations.


The Poincaré disk model, another representation of hyperbolic geometry, maps the entire hyperbolic plane onto the interior of a unit disk. In this model, geodesics are represented by arcs of circles that are orthogonal to the boundary of the disk. This representation is useful for visualizing hyperbolic space and for solving problems involving hyperbolic distances and angles. The Poincaré disk model preserves angles but distorts distances, providing a different but complementary perspective to the Weierstrass coordinates.

.. automodule:: hypertiling.representations
    :members: