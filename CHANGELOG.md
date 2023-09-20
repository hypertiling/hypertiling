## v1.3.1
Date: 2023-09-20

Today, we are updating the build system of hypertiling, simultaneously with the publication of our release manuscript on the arXiv preprint server. As a user, you should not notice any changes as a result of this update. Owed to the new configuration, hypertiling can now be installed with numba using

```
pip install hypertiling[numba]
```
or
```
pip install .[numba]
```

for a local installation, after the latest version has been cloned from our repository.

**Further changes**

- bug fixes
- improved pipeline
- more unit tests
- introduction of test coverage

## v1.3
Date: 2023-08-28


**Release Notes**

An important milestone in the development of algorithms for the construction of hyperbolic tilings is the combinatorial method presented by Douglas Dunham and coworkers in a series of publications in the early 1980s [1,2]. With hypertiling v1.3, we not only bring this influential algorithm into the package, but also provide modern implementations, based on Ref. [3]. Specifically, the new features are:


- New kernel: **DUN07 (Dunham)**: An modern implementation of the construction algorithm by Douglas Dunham. Recursive calls to a hierarchical tree structure are used to build duplicate free tilings in hyperboloid coordinates rather than in the Poincare disk representation.

- New kernel: **DUN07X (DunhamX)**: A heavily optimized variant of `DUN07`, with a performance increase of more than one order of magnitude.

- The previously existing, experimental implementation of Dunham's algorithm becomes deprecated and is renamed to **DUN86 (DunhamLegacy)**. It will be removed in one of the following releases.

Owed to the construction mechanism, the `get_layer` function is not available for tilings created by the two new kernels.


**Further changes**

- several bug fixes across the package
- minor performance optimizations
- improved documentation
- new example notebooks
- auto-import submoduls

References
 > [1] D. Dunham, J. Lindgren and D. Witte, Creating repeating hyperbolic patterns, In Pro-
ceedings of the 8th annual conference on Computer graphics and interactive techniques,
pp. 215–223 (1981)

> [2] D. Dunham, Hyperbolic symmetry, In Symmetry, pp. 139 – 153. Pergamon (1986)

> [3] D. Dunham, An algorithm to generate repeating hyperbolic patterns, the Proceedings of
ISAMA pp. 111–118 (2007).

## v1.2
Date: 2023-03-23


**Release Notes**

In hypertiling v1.2 we are proud to introduce a number of exciting new features

- The **hyperbolic graph**:<br> What is the difference between a `HyperbolicTiling` and a `HyperbolicGraph`? In many application one is interested in the adjacency relations between the cells of a tiling, rather than in their coordinates. Neighborhood relations can be constructed from an existing tiling using the `get_nbrs_list` method. However, in order to avoid this detour, we introduce kernels which construct **only** the graph structure of a hyperbolic lattice, independent from a coordinate representation of cells. These objects yield reduced features and functionality compared to a full `HyperbolicTiling`. A tiling on the other hand _can_, but not necessarily needs to provide graph relations between its individual cells. Even though tilings and graphs are different objects, neighbors are accessed in the same way, using the following two methods

```python
   # let T be a HyperbolicTiling or HyperbolicGraph
   T.get_nbrs(i)   # return neighbors of cell i
   T.get_nbrs_list()   # return neighbors of all cells
```

- New kernel: **GRG (generative reflection graph)**: <br>This is a variant of the GR kernel, which constructs neighborhood relations already during the construction of the lattice. Cell coordinates are not stored. It is hence a `Graph` kernel. The kernel follows the GR construction principles. Only one symmetry sector is explicitly stored, whereas any information outside this sector will be generated on demand.

- New kernel: **GRGS (generative reflection graph static)**:<br> This is a static variant of the GRG kernel. Adjacency relations for all cells are stored explicitly. No sector construction and and no on-demand generation is used. Hence the memory requirement is about a factor p larger compared to GRG. Nonetheless GRGS is still blazing fast and therefore particularly suited for large-scale scientific simulations of systems with local interactions.

- New kernel: **SRG (static rotational graph)**:<br> Currently in a testing state, this kernel will become the default kernel in a future release. It uses the construction principle of the SR (static rotational) kernel family and is able to manipulate existing lattices by dynamically adding, removing and filter cells. A demonstration notebook is already available. More coming soon!

- Rename SR (static rotational) kernel to SRS (static rotational sector), which becomes necessary due to the introduction of SRG

- hypertiling now supports **verbosity levels** for printed output. Available rules are "Warning", "Status", "Debug" and "Develop", where "Warning" is set default. The verbosity can be adjusted globally using the `set_verbosity_level` helper function

- Kernels can now be selected by their abbreviation as well as by their class name, e.g.:

```python
   from hypertiling import TilingKernels
   T = HyperbolicTiling(7,3,2, kernel=TilingKernels.GenerativeReflection)
```

**Further changes**

- minor fixes in the GR kernel
- improved unit tests
- adjusted kernel class names (remove Kernel... prefix)
- adjusted file names
- tidied up abstract base classes
- improved naming of utility functions
- Tilings and Graphs are required to implement __len__ method


**Coming up soon**

- SRG kernel feature presentation
- path animations!
- new example notebooks


## v1.1.3
Date: 2022-12-02

**Release Notes**

This release includes several performance and stability optimizations of the static rotational (SR) kernel, as well as a clearer internal code design regarding the duplicate containers. Most important effects of these changes are:

- the static rotational improved (SRI) kernel is from now on only labelled SR. It remains the default kernel.
- the former static rotational (SR) kernel is now deprecated and labelled as SRL (legacy). It will be removed in one of the upcoming releases.

**Further changes**

- performance optimizations in the GR kernel
- get_nbrs method for SR kernel was missing
- updated demo notebooks
- several bugfixes



## v1.1.2
Date: 2022-11-02

**Release Notes**

- improved SVG drawing capabilites (now more flexible, robust and compatible with GR kernel; introduce SVG string class)
- debug and clean up Jupyter demo notebooks
- path animations are now available (in a testing state)
- hypertiling now has a CHANGELOG file
- several bugfixes


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
