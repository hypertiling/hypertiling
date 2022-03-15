<table  align="center"><td align="center" width="9999">

<img src="assets/logo_frameless_font.svg" align="center" width="300" alt="project icon">

</td>
<tr>
<td align="left" width="9999" >



**hypertiling** is a Python 3 libary for the fast generation of regular hyperbolic tilings, embedded in the Poincare disk model.

Using efficient algorithms and the power of numpy, hyperbolic graphs with millions of polygons can be created in a matter of minutes on a single CPU. We also provide optimized search algorithms for finding adjacent vertices, which allows to use the graph for all sorts of scientific purposes.

# Installation

Hypertiling is not yet available for `pip`, but can be locally built on Linux systems using
```
$ python setup.py install
```

For developer mode use
```
$ python setup.py develop
```

# Usage

Import tiling object from *hypertiling* library

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

Further information can be found in our Jupyter notebooks in /examples subfolder. 

# Authors
* Manuel Schrauth (mschrauth@physik.uni-wuerzburg.de)
* Felix Dusel

This project is developed at: <br>
Institute for Theoretical Physics and Astrophysics<br>
University of Wuerzburg


<p align="center">
  <img src="assets/hyp6.svg" width="400" />
</p>



# License
Every part of hypertiling is available under the MIT license.

