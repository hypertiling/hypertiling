import inspect
import warnings

try:
    from numba import njit

    AVAILABLE = True
except Exception as error:
    warnings.warn("Failed to import numba... Use non-numba mode")
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


class DelayChecker:
    """
    This is just a workaround for the recursive ahead of time compilation as this seems not to be possible till now
    https://numba.discourse.group/t/how-to-compile-ahead-of-time-a-recursive-function/1630
    """

    def __init__(self, signature, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.signature = signature
        self.compiled = False

    def __call__(self, f):
        if AVAILABLE:
            return Wrapper(f, self.signature, *self.args, **self.kwargs)
        else:
            return f


class Wrapper:
    """
    This is just a workaround (together with DelayChecker) for the recursive ahead of time compilation as this seems
    not to be possible till now
    https://numba.discourse.group/t/how-to-compile-ahead-of-time-a-recursive-function/1630
    """

    def __init__(self, f, signature, *args, **kwargs):
        self.compiled = False
        self.f = f
        self.signature = signature
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self.compiled is False:
            self.f = njit(self.signature, *self.args, **self.kwargs, cache=False)(self.f)
            self.compiled = True
        return self.f(*args, **kwargs)
