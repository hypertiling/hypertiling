<table  align="center"><td align="center" width="9999">

<img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/logo/logo73.svg" align="center" width="380" alt="project icon">

</td>
<tr>
<td align="left" width="9999" >




![PyPI](https://img.shields.io/pypi/v/hypertiling)
![Discord](https://img.shields.io/discord/990718743455883336?label=discord)
![Website](https://img.shields.io/website?down_message=offline&up_message=online&url=http%3A%2F%2Fwww.hypertiling.de%2F)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/hypertiling)](https://pypistats.org/packages/hypertiling)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7559394.svg)](https://doi.org/10.5281/zenodo.7559394)


**hypertiling** is a high-performance Python 3 library for the generation and visualization of hyperbolic lattices, embedded in the Poincare disk model. Using efficient algorithms and the CPU/SIMD optimization provided by numpy, hyperbolic tilings with millions of vertices can be created in a matter of minutes on a single workstation computer. Facilities including optimized search algorithms for adjacent vertices and powerful plotting and animation capabilities are provided to support scientific and other advanced uses of the graphs.


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
* Felix Dusel
* Florian Goth
* Dietmar Herdt
* Jefferson S. E. Portela

This project is developed at:  
[Institute for Theoretical Physics and Astrophysics](https://www.physik.uni-wuerzburg.de/en/tp3/home/)  
[University of Wuerzburg](https://www.uni-wuerzburg.de/en/home/)

## Examples

Simulation of a Ising-like Boltzmann spin model on a hyperbolic (7,3) tiling at low temperature. One readily recognizes a number of magnetic domains of opposite spin orientation (red/blue areas). The domain walls are approximately given by arcs of Euclidean circles contained within the disk and orthogonal to its boundary. These arcs represent straight lines in the Poincare disk representation of the hyperbolic plane.

<p align="center">                                                                                                                                                                                                                           
  <img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/hyp6.svg" width="300" />                                                                                                                         
</p>



Further information and examples can be found in our Jupyter notebooks in /examples subfolder. 


## License
Every part of hypertiling is available under the MIT license.
