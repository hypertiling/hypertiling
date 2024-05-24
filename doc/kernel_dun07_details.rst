Kernel Details
--------------

The kernel features a number of methods that can be used to construct the tiling:
First we define a method to create the vertices of a fundamental hyperbolic (p,q) polygon. It calculates the radius, constructs the vertices using trigonometric functions, and then transforms the vertices to Weierstrass coordinates.

.. autofunction:: hypertiling.kernel.DUN07.Dunham._create_fundamental_polygon

Now we implement a recursive function that implements the Dunham tiling algorithm. It replicates a polygonal pattern by iterating over vertices and polygons, updating transformations and exposures, and recursively calling itself to draw the pattern at different layers of recursion.

.. autofunction:: hypertiling.kernel.DUN07.Dunham._replicate_motif

The method _replicate is the driver routine for the Dunham tiling algorithm. It takes a polygonal object as a parameter and initiates the drawing of the second layer, starting the recursion. If the number of desired layers is 1, it returns immediately. Otherwise, it iterates over each vertex of the polygon and, for each vertex, iterates over a certain number of polygons. For each polygon, it determines the exposure level, replicates the motif, and modifies the transformation.

.. autofunction:: hypertiling.kernel.DUN07.Dunham._replicate

What is left is the function which initiates the construction:

.. autofunction:: hypertiling.kernel.DUN07.Dunham._generate

