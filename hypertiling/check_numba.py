import inspect
import warnings

try:
    from numba import njit

    AVAILABLE = True
except:
    AVAILABLE = False


class NumbaChecker:

    def __init__(self, signature=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.signature = signature

    def __call__(self, f):
        if AVAILABLE:
            if not (self.signature is None):
                return njit(self.signature, *self.args, **self.kwargs)(f)
            else:
                warnings.warn(
                    f"{f.__name__} in {inspect.getmodule(f)}:\n\tNo signature specified. Use lazy compilation instead!")
                return njit(f)
        else:
            return f


def check_numba(f):
    if AVAILABLE:
        warnings.warn("check_numba is outdated and will be removed soon. Use NumbaChecker instead!")
        return njit(f)
    else:
        return f
