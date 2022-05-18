<table  align="center"><td align="center" width="9999">

<img src="https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling/-/raw/master/assets/logo/logo73.svg" align="center" width="380" alt="project icon">

</td>
<tr>
<td align="left" width="9999" >



**hypertiling** is a Python 3 libary for fast generation of regular hyperbolic tilings, embedded in the Poincare disk model.

Using efficient algorithms and the power of numpy, hyperbolic graphs with millions of polygons can be created in a matter of minutes on a single CPU. We also provide optimized search algorithms for finding adjacent vertices, which allows to use the graph for all sorts of scientific purposes.

## Source

The package can be found and downloaded in our public [git repository](https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling).


## Installation and Usage

Hypertiling is available in the [PyPI](https://pypi.org/) package index and can be installed using
```
$ pip install hypertiling
```
The package can also be locally installed. First download or clone the package, using
```
$ git clone https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling
```
Now execute 
```
$ pip install .
```
in the package's root directory to install the package in-place.

For developer mode use
```
$ pip install -e .
```


In Python, import tiling object from *hypertiling* library

```python
from hypertiling import HyperbolicTiling
```
Set parameters, initialize and generate the tiling

```python
p = 7
q = 3
nlayers = 5

T = HyperbolicTiling(p,q,nlayers) 
T.generate()
```


## Authors
* Manuel Schrauth  
mschrauth@physik.uni-wuerzburg.de
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
