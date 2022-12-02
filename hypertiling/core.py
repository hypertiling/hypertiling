# relative imports
from .static.static_rotational import KernelStaticRotational
from .static.static_rotational_legacy import KernelStaticRotationalLegacy
from .static.legacy_dunham import KernelLegacyDunham
from .generative.generative_reflection import KernelGenerativeReflection
from .ion import htprint

KERNELS = {"SR": KernelStaticRotational,
           "SRL":KernelStaticRotationalLegacy,
           "DUN": KernelLegacyDunham,
           "GR": KernelGenerativeReflection}


# factory pattern allows to select between kernels
def HyperbolicTiling(p, q, n, center="cell", kernel="SR", verbose=False, **kwargs):
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
        return KERNELS[kernel](p, q, n, **kwargs)

    elif kernel == "SR" or kernel == "SRL":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        return KERNELS[kernel](p, q, n, center, **kwargs)

    elif kernel == "DUN":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        htprint("Warning", "Dunham kernel is only implemented for legacy reasons and largely untested. Use with care!")
        if center == "vertex":
            htprint("Warning", "Dunham kernel does not support vertex centered tilings yet!")
        return KERNELS[kernel](p, q, n, center, **kwargs)     

    #elif ... (further kernels)
    
    else:
        raise KeyError("[hypertiling] Error: No valid kernel specified")

    
    

