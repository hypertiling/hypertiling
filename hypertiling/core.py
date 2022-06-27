# relative imports
from .kernelflo import KernelFlo
from .kernelmanu import KernelManu
from .kerneldunham import KernelDunham




# factory pattern allows to select between kernels
def HyperbolicTiling(p, q, n, center="cell", kernel="flo"):
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

    if (p-2)*(q-2) <= 4:
        raise AttributeError("[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p>20 or q>20 and n>5:
        print("[hypertiling] Warning: The lattice might become very large with your parameter choice!")


    kernels = { "manu":   KernelManu, # to-do: we need better names for the kernels ;)
                "flo":    KernelFlo, 
                "dunham": KernelDunham}

    if kernel not in kernels:
       raise KeyError("[hypertiling] Error: No valid kernel specified")
    if kernel == "dunham":
        print("Caution: This kernel is slow and error-prone. Use at own risk!")
        if center == "vertex":
            print("KernelDunham doesn't support vertex-centered tilings yet. A cell-centered tiling will be generated instead...")
        # raise NotImplementedError("[hypertiling] Error: Dunham kernel is currently broken (fixme!)")
    return kernels[kernel](p, q, n, center)
