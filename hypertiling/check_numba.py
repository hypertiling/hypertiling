from typing import get_type_hints
import inspect
import warnings

try:
    from numba import njit

    AVAILABLE = True
except:
    AVAILABLE = False


class NumbaChecker:

    def __init__(self, *args):
        self.args = args

    def __call__(self, f):
        if AVAILABLE:
            if self.args:
                return njit(f, *self.args)
            else:
                warnings.warn("No signature specified. Use lazy compilation instead!")
                return njit(f)
        else:
            return f


def check_numba(f):
    if AVAILABLE:
        warnings.warn("check numba is outdated and will be removed soon. Use NumbaChecker instead!")
        return njit(f)
    else:
        return f
