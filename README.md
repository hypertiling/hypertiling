<table  align="center"><td align="center" width="9999">

<img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/logo/logo73.svg" align="center" width="380" alt="project icon">

</td>
<tr>
<td align="left" width="9999" >

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/hypertiling)](https://pypi.org/project/hypertiling/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/hypertiling)](https://pypistats.org/packages/hypertiling)
[![Website](https://img.shields.io/website?down_message=offline&up_message=online&url=http%3A%2F%2Fwww.hypertiling.de%2F)](http://www.hypertiling.de)
[![Discord](https://img.shields.io/discord/990718743455883336?label=discord)](https://discord.gg/f9GW9B2Ezs)<br>
<a href="https://doi.org/10.5281/zenodo.7559393">
  <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7559393.svg" alt="DOI">
</a>
[![badge_coverage][]]()
[![badge_pipeline][]](https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/pipelines)

</div>




[badge_coverage]: https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/badges/master/coverage.svg
[badge_maintainability]: https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/jobs/298389/artifacts/raw/public/badges/maintainability.svg
[badge_pipeline]: https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/badges/master/pipeline.svg


**hypertiling** is a high-performance Python library for the generation and visualization of regular hyperbolic lattices embedded in the Poincare disk model. Using highly optimized, efficient algorithms, hyperbolic tilings with millions of vertices can be created in a matter of minutes on a single workstation computer. Facilities including computation of adjacent vertices, dynamic lattice manipulation, refinements, as well as powerful plotting and animation capabilities are provided to support advanced uses of hyperbolic graphs. 


## Installation

hypertiling is available in the [PyPI](https://pypi.org/) package index and can be installed using
```
$ pip install hypertiling
```

For optimal performance, we highly recommand to use hypertiling together with `python-numba`, which, if not already present on your system can be installed automatically using the `[numba]`-suffix, i.e.
```
$ pip install hypertiling[numba]
```
The package can also be locally installed from our public [git repository](https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling) via
```
$ git clone https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling
$ pip install .
```


## Quick Start

In Python, import tiling object from the **hypertiling** library

```python
from hypertiling import HyperbolicTiling
```
Set parameters, initialize and generate the tiling

```python
p = 7
q = 3
nlayers = 5

T = HyperbolicTiling(p,q,nlayers) 
```

## Documentation

Further usage examples and a full API reference are available in our [documentation](https://gitpages.physik.uni-wuerzburg.de/hypertiling/hyperweb/doc/examples/quickstart.html).

## Authors
* Manuel Schrauth  
manuel.schrauth@iis.fraunhofer.de
* Yanick Thurn
* Florian Goth
* Jefferson S. E. Portela
* Dietmar Herdt
* Felix Dusel

This project is developed at:  
[Institute for Theoretical Physics and Astrophysics](https://www.physik.uni-wuerzburg.de/en/tp3/home/)  
[University of Wuerzburg](https://www.uni-wuerzburg.de/en/home/)



## Development

This project uses [Hatch](https://hatch.pypa.io/latest/) to manage environments and [pre-commit](https://pre-commit.com/) to keep the repository history clean.

### 1. Environment Setup
To initialize the development environment and all dependencies:
```bash
pip install hatch
hatch shell
```
**Important:** Upon first run, Hatch automatically installs a **git pre-commit hook**. This hook silently strips execution counts and cell outputs (like plots and heavy binary data) from your `.ipynb` files whenever you `git commit`. This ensures the repository remains lightweight and provides clean, readable code diffs.

### 2. Working with Examples
The notebooks in `examples/` are stored "clean" (without outputs). To populate them with results and plots on your local machine, run:
```bash
hatch run examples
```


### 3. Documentation Build
The documentation includes the rendered notebooks. To build the full Sphinx site locally:
```bash
hatch run doc:build
```
This command executes the notebooks in a temporary environment and generates the HTML at `doc/_build/html/index.html`.




## Citation

If you use _hypertiling_, we encourage you to cite or reference this work as you would any other scientific research. The package is a result of a huge amount of time and effort invested by the authors. Citing us allows us to measure the impact of the research and encourages others to use the library.

Codebase:
> Manuel Schrauth, Yanick Thurn, Florian Goth, Jefferson Portela, Dietmar Herdt and Felix Dusel. (2023). The _hypertiling_ project. Zenodo. https://doi.org/10.5281/zenodo.7559393

Release publication:
> Manuel Schrauth, Yanick Thurn, Florian Goth, Jefferson Portela, Dietmar Herdt and Felix Dusel. (2024). HYPERTILING - a high performance python library for the generation and visualization of hyperbolic lattices project. [SciPost Physics Codebases (2024): 034.](https://www.scipost.org/SciPostPhysCodeb.34)

Further publications:
> Yanick Thurn, Manuel Schrauth, Johanna Erdmenger. Hyperbolic tiling neighborhoods in O(1) time. [arXiv preprint](https://arxiv.org/abs/2508.04765)

## Examples


### Tilings

The core functionality of the package is the generation of regular hyperbolic tilings projected onto the Poincare disk. Tilings in the upper row are centered about a cell, in the lower row about a vertex.


<p align="center">   
 <img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/tilings.png" width="700" />   
</p>



### Refinements

The hypertiling package allows to perform triangle refinements, such as shown here

<p align="center">   
 <img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/refinments.png" width="700" />   
</p>




## Applications

### Hyperbolic Magnet
Simulation of a Ising-like Boltzmann spin model with anti-ferromagnetic interactions on a hyperbolic (7,3) tiling, quenched at low temperature. The hyperbolic antiferromagnet (left) exhibits geometrical frustration, whereas on a flat lattice (right) an ordered anti-parallel alignment can be observed.
<p align="center">                                                                                                                                                                                                                           
  <img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/magnet.png" width="900" />                                                                                                                         
</p>


### Helmholtz Equation
Solution of an electrostatic Helmholtz problem on a refined (3,7) tiling, where boundary values have been fixed to either -1 (red) or +1 (blue). One readily recognizes a field value separation according to geodesic arcs in the Poincare disk representation of the hyperbolic plane.


<p align="center">                                                                                                                                                                                                                           
  <img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/helmholtz.png" width="450" />                                                                                                                         
</p>


Further information and examples can be found in our Jupyter notebooks in /examples subfolder. 


## License
Every part of hypertiling is available under the MIT license.
