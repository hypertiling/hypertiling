# relative imports
from .kernelflo import KernelFlo
from .kernelmanu import KernelManu
from .kerneldunham import KernelDunham




# factory pattern allows to select between kernels
def HyperbolicTiling(p, q, n, center="cell", kernel="manu"):
    """
    The base function which invokes a hyperbolic tiling

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    center : str
        decides whether the tiling is constructed about a "vertex" or "cell" (default)
    kernel : str
        selects the construction algorithm
    """


    kernels = { "manu":   KernelManu, # to-do: we need better names for the kernels ;)
                "flo":    KernelFlo, 
                "dunham": KernelDunham}
    if kernel not in kernels:
       raise KeyError("no valid kernel specified")
    if kernel == "dunham":
        raise NotImplementedError("Dunham kernel is currently broken (fixme!)")
    return kernels[kernel](p, q, n, center)
