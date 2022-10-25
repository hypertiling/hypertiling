## v1.1.1
Date: 2022-10-25


**Release Notes**

- **Autogeneration**: Tilings are now immediately constructed after their initilization, resulting in a one-liner `T = HyperbolicTiling(7,3,4)`. The additional generate method which was formerly required by some kernels to construct the tiling after the initialization, is now executed on default.
- Autogeneration keyword argument: In case an autogeneration is not intended, it can be switched off using the new keyword argument `T = HyperbolicTiling(7,3,4,autogenerate=False)`
- Unify names of kernel-specific neighbour functions: They all go under the prefix "get_nbrs" now
- More consistent naming convention of plot functions, such as `convert_edges_to_arcs` and `convert_polygons_to_patches`
- Providing a caching option improves library import speed
- Clean up transformations
- Several bugfixes


## v1.1
Date: 2022-10-04

**Release Notes**
- We are proud to present the **Generative Reflection (GR) kernel**. Owing to a novell, sophisticated tiling construction algorithm and its intrinsic generative nature, where only one symmetry sector of the tiling is held stored, the GR kernel is extremely fast and at the same time significantly less memory consuming compared to our existing kernels.
- **New kernel API**: Introducing the GR kernel made is necessary to establish a number of new interfaces, which can be used to access tiling parameters. 
- **Homogenized graph routines**: Since every kernel comes with it's own graph/neighbour routines, we homogonized those interfaces, too. Additionally, the refactored `find` routine in the neighbour module can be used with any kernel.
- hyperanimator: currently in a testing state, this class provides a simple way to realize **animated plots** on hyperbolic lattices. Watch out for even more animation features in upcoming releases

**Further Changes**
- renaming existing kernels to static rotational (SR) and static rotational improved (SRI)
- introduction of abstract kernel base class
- several bugfixes in Dunham legacy kernel (DUN)
- introduction of numba dectorator in order to reduce code complexity
- new unit tests
- several bugfixes
