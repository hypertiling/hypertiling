from .ion import htprint
from .kernels.SRG import KernelStaticRotationalGraph
from .kernels.SR  import KernelStaticRotational
from .kernels.SRL import KernelStaticRotationalLegacy
from .kernels.DUN import KernelLegacyDunham
from .kernels.GR  import KernelGenerativeReflection
from .kernels.GRG import KernelGenerativeReflectionGraph
from .kernels.SGRG import KernelStaticReflectionGraph


TILINGS = { "SR":  KernelStaticRotational,
            "SRG": KernelStaticRotationalGraph,
            "SRL": KernelStaticRotationalLegacy,
            "DUN": KernelLegacyDunham,
            "GR":  KernelGenerativeReflection  }

GRAPHS = {  "GRG": KernelGenerativeReflectionGraph,
            "SGRG": KernelStaticReflectionGraph}


def HyperbolicTiling(p, q, n, kernel="SR", **kwargs):
    """
    The factory pattern function which invokes a hyperbolic tiling

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : str
        selects the construction algorithm
    """

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError("[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == "SRL":
        htprint("Warning", "This kernel is deprecated! Better use the 'SR' kernel instead!")
    if kernel == "GR":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
    if kernel in ["SR", "SRG", "SRL"]:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
    if kernel == "DUN":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        htprint("Warning", "Dunham kernel is only implemented for legacy reasons and largely untested. Use with care!")

    if kernel not in TILINGS:
        raise KeyError("[hypertiling] Error: No valid kernel specified")

    return TILINGS[kernel](p, q, n, **kwargs)



def HyperbolicGraph(p, q, n, kernel="SR", **kwargs):
    """
    The factory pattern  function which invokes a hyperbolic graph

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : str
        selects the construction algorithm
    """

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError("[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == "GRG":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        
    if kernel not in GRAPHS:
        raise KeyError("[hypertiling] Error: No valid kernel specified")

    return GRAPHS[kernel](p, q, n, **kwargs)