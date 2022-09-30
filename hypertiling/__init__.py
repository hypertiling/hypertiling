from .core import HyperbolicTiling

__all__ = ['HyperbolicTiling']

__version__ = "1.0.2"
__author__ = 'Manuel Schrauth, Felix Dusel, Florian Goth, Dietmar Herdt, Jefferson S. E. Portela, Yanick Thurn'
__credits__ = 'Institute for Theoretical Physics and Astrophysics, University of Wuerzburg'
__packages__ = []


# for njiting the functions
import sys
import io

__std_out = sys.stdout
sys.stdout = io.StringIO()

tiling = HyperbolicTiling(3, 7, 2, kernel="GR")

sys.stdout = __std_out