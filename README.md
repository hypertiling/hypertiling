<table  align="center"><td align="center" width="9999">

<img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/logo/logo73.svg" align="center" width="380" alt="project icon">

</td>
<tr>
<td align="left" width="9999" >




![PyPI](https://img.shields.io/pypi/v/hypertiling)
![Discord](https://img.shields.io/discord/990718743455883336?label=discord)
![Website](https://img.shields.io/website?down_message=offline&up_message=online&url=http%3A%2F%2Fwww.hypertiling.de%2F)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/hypertiling)](https://pypistats.org/packages/hypertiling)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7559393.svg)](https://doi.org/10.5281/zenodo.7559393)


**hypertiling** is a high-performance Python library for the generation and visualization of regular hyperbolic lattices embedded in the Poincare disk model. Using highly optimized, efficient algorithms, hyperbolic tilings with millions of vertices can be created in a matter of minutes on a single workstation computer. Facilities including computation of adjacent vertices, dynamic lattice manipulation, refinements, as well as powerful plotting and animation capabilities are provided to support advanced uses of hyperbolic graphs. 


## Installation

hypertiling is available in the [PyPI](https://pypi.org/) package index and can be installed using
```
$ pip install hypertiling
```
The package can also be locally installed from our public [git repository](https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling) via
```
$ git clone https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling
$ pip install .
```
## Usage

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

Further examples are available in our [documentation](https://gitpages.physik.uni-wuerzburg.de/hypertiling/hyperweb/doc/examples/quickstart.html) and Jupyter demo notebooks.

## Authors
* Manuel Schrauth  
mschrauth@physik.uni-wuerzburg.de
* Yanick Thurn
* Florian Goth
* Jefferson S. E. Portela
* Dietmar Herdt
* Felix Dusel

This project is developed at:  
[Institute for Theoretical Physics and Astrophysics](https://www.physik.uni-wuerzburg.de/en/tp3/home/)  
[University of Wuerzburg](https://www.uni-wuerzburg.de/en/home/)

## Citation

If you use _hypertiling_, we encourage you to cite or reference this work as you would any other scientific research. The package is a result of a huge amount of time and effort invested by the authors. Citing us allows us to measure the impact of the research and encourages others to use the library.

Cite us:

> Manuel Schrauth, Yanick Thurn, Florian Goth, Jefferson S.E. Portela, Dietmar Herdt and Felix Dusel. (2023). The _hypertiling_ project. Zenodo. https://doi.org/10.5281/zenodo.7559393



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
