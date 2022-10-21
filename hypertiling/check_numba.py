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
                return njit(self.signature, *self.args, **self.kwargs, cache=True)(f)
            else:
                warnings.warn(
                    f"{f.__name__} in {inspect.getmodule(f).__file__}:\n" + \
                    f"\tNo signature specified. Use lazy compilation instead!")
                return njit(f, cache=True)
        else:
            return f
