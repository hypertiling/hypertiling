from .ion import htprint
from .kernels.SRG import KernelStaticRotationalGraph
from .kernels.SR  import KernelStaticRotational
from .kernels.SRL import KernelStaticRotationalLegacy
from .kernels.DUN import KernelLegacyDunham
from .kernels.GR  import KernelGenerativeReflection
from .kernels.GRG import KernelGenerativeReflectionGraph


TILINGS = { "SR":  KernelStaticRotational,
            "SRG": KernelStaticRotationalGraph,
            "SRL": KernelStaticRotationalLegacy,
            "DUN": KernelLegacyDunham,
            "GR":  KernelGenerativeReflection}

GRAPHS = {
            "GRG": KernelGenerativeReflectionGraph}


def HyperbolicTiling(p, q, n, center="cell", kernel="SR", **kwargs):
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

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError(
            "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if "radius" in kwargs and kwargs["radius"] is not None:
        htprint("Status", "You have defined a cut-off radius ... make sure you set n large enough ...")

    if kernel == "SRL":
        htprint("Warning", "This kernel is deprecated! Better use the 'SR' kernel instead!")

    if kernel == "GR":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        return TILINGS[kernel](p, q, n, **kwargs)

    elif kernel in ["SR", "SRG", "SRL"]:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        return TILINGS[kernel](p, q, n, center, **kwargs)

    elif kernel == "DUN":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        htprint("Warning", "Dunham kernel is only implemented for legacy reasons and largely untested. Use with care!")
        if center == "vertex":
            htprint("Warning", "Dunham kernel does not support vertex centered tilings yet!")
        return TILINGS[kernel](p, q, n, center, **kwargs)

        # elif ... (further kernels)

    else:
        raise KeyError("[hypertiling] Error: No valid kernel specified")


def HyperbolicGraph(p, q, n, center="cell", kernel="SR", verbose=False, **kwargs):
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

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError(
            "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == "GRG":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        return GRAPHS[kernel](p, q, n, **kwargs)

    else:
        raise KeyError("[hypertiling] Error: No valid kernel specified")